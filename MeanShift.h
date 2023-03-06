#pragma once

#include <cstring> // memcpy()
#include <cmath>   // std::exp(),::pow()

// OpenMP parallelization has currently only very slight benefits for the small
// datasets/dimensionality this code is designed for, partly because parfor
// comes at the cost of additional memory allocation of otherwise shared buffers.
#define MEANSHIFT_ENABLE_OMP

/**
  Mean-shift clustering with adaptive bandwidth.
  
  This is a brute force implementation w/o discretization for efficient
  processing of small datasets with n<1500 points of up to d<8 dimensions.
  The class is self-contained, i.e. has no additional dependencies beyond std,
  operates on raw data pointers and is header-only.

  The main routine is shift_points() that implements mean-shift of a set of
  test points against a density estimate of another set of data points. Note
  that the routine does not compute a clustering/segmentation but outputs only
  the mean-shifted test points that, in case the mean-shift algorithm has
  converged, will accumulate at candidate positions for maximum modes of 
  the density. Identification of stable modes (e.g. via perturbation analysis)
  and deriving a clustering from the result is up to the callee.

  \author Max Hermann
  \date Nov 2015
*/
template<class T>
class MeanShift
{
public:
    /// Mean-shift options (so far only used internally, see c'tor arguments)
	struct Options
	{
		T        epsilon;
		unsigned maxIter;
        bool     doStats;
		
        Options()
        : epsilon( 0.01 ),
          maxIter( 100 ),
          doStats( false )
        {}

		Options( T epsilon_, unsigned maxIter_ )
		: epsilon( epsilon_ ),
		  maxIter( maxIter_ ),
          doStats( false )
		{}		
	};
	
    /** C'tor
        @param[in] epsilon  convergence threshold on shift distance 
            (choose as some tiny percentage of bandwidth, impacts performance).
        @param[in] maxIter  maximum number of iterations 
            (default value of 100 should usually be sufficient).
    */
	MeanShift( T epsilon, unsigned maxIter=100 )
	: m_opts( epsilon, maxIter ),
      m_stat_iters( nullptr )
	{}

    ~MeanShift()
    {
        if( m_stat_iters ) delete [] m_stat_iters;
    }

    ///@name Statistics about last mean-shift execution
    ///@{
    /// Enable or disable statistics
    void enable_stats( bool b=true ) { m_opts.doStats = b; }
    /// Get array with convergence info per point.
    /// -1=maxiter hit, -2=mean-shift failed, >0=number of iterations until convergence
    int* stat_iters() { return m_stat_iters; }
    ///@}	


	/** Adaptive bandwidth non-blurring mean shift procedure applied to the m data points in Y.
		@param[in] X          n x d data matrix of n points of dimension d
		@param[in] bandwidth  vector with n scale parameters, one for each point
		@param[in] n          number of data points in X
		@param[in] d          dimension of data points
		@param[in,out] Y      pre-allocated m x d data matrix of test points to which mean-shift is applied (in-place)
		@param[in] m          number of test points in Y
	*/
	void shift_points( const T* X, const T* bandwidth, unsigned n, unsigned dim, T* Y, unsigned m )    
    {
	    T* Y0        = new T[m*dim];
        T* norm      = new T[n];
#ifndef MEANSHIFT_ENABLE_OMP
        // Globally shared buffers when not working in parallel
	    T* distances = new T[n];
	    T* weights   = new T[n];
#endif

        if( m_opts.doStats )
        {
            if( m_stat_iters ) delete [] m_stat_iters;
            m_stat_iters = new int[m];
        }

        // Pre-compute normalization
        compute_adaptive_normalization( bandwidth, n, dim, norm );
	
	    // Iterate all points in Y
	    memcpy( (void*)Y0, (void*)Y, m*dim*sizeof(T) );
#ifdef MEANSHIFT_ENABLE_OMP
        #pragma omp parallel for  
#endif
	    for( int i=0; (unsigned)i < m; ++i )
	    {
#ifdef MEANSHIFT_ENABLE_OMP
            // Local buffers when working in parallel
	        T* distances = new T[n];
	        T* weights   = new T[n];
#endif

            if( m_opts.doStats ) m_stat_iters[i] = -1;

		    T* y0i = &Y0[i*dim];
		    T* yi  = &Y [i*dim];
		
		    unsigned iter=0;
		    bool converged=false;
		    while( iter < m_opts.maxIter && !converged)
		    {
			    // Mean shift
			    dist2( y0i, X, n,dim, distances );	               // d_i  = ||y_i - x_i||^2
			    normal_kernel( distances, n, bandwidth, weights ); // w'_i = exp( -d_i / (h_i*h_i) )
                multiply_pointwise( weights, norm, n, weights );   // w_i  = (1 / h_i^(d+2)) * w'_i
                if( !eval_meanshift( X, weights, n,dim, yi ) )     // y_j = sum_i( x_i*w_i ) / sum_i( w_i )
                {
                    // Mean-shift could not be computed!
                    // This may happen in case of zero weights, i.e. test points 
                    // are too far off the estimated density or bandwidth is chosen
                    // way too small.

                    // Skip iteration for this point.
                    if( m_opts.doStats ) m_stat_iters[i] = -2;
                    break;
                }

			    // Check for convergence
			    T delta;
			    dist2( y0i, yi, 1,dim, &delta );
			    if( sqrt(delta) < m_opts.epsilon )
			    {
				    converged = true;

                    if( m_opts.doStats ) m_stat_iters[i] = iter;
			    }
			
			    // Update
			    memcpy( (void*)y0i, (void*)yi, dim*sizeof(T) );
			    iter++;
		    }

#ifdef MEANSHIFT_ENABLE_OMP
	        delete [] weights;
	        delete [] distances;
#endif
	    }
	
#ifndef MEANSHIFT_ENABLE_OMP
	    delete [] weights;
	    delete [] distances;
#endif
        delete [] norm;
	    delete [] Y0;
    }
		
	/// Adaptive bandwidth non-blurring mean shift procedure applied to *all* data points in X.		
	void shift_all( const T* X, const T* bandwidth, unsigned n, unsigned dim, T* Y )
    {
	    memcpy( (void*)Y, (void*)X, n*dim*sizeof(T) );
        shift_points( X, bandwidth, n, dim, Y, n );
    }

	/** Pilot density estimation with constant global bandwidth h0.
	    This can be useful to derive an adaptive bandwidth per point.
		@param[in] X          n x d data matrix of n points of dimension d
		@param[in] h0         constant global bandwidth h0 for pilot study  
		@param[in] n          number of data points in X
		@param[in] d          dimension of data points
		@param[out] density   pre-allocated 1 x n matrix for result density
	*/	
	static void pilot_density_estimate( const T* X, T h0, unsigned n, unsigned dim, T* density )
	{		
#ifndef MEANSHIFT_ENABLE_OMP
        // The sqdist buffer can only be re-used if not working in parallel
	    T* sqdist  = new T[n];
#endif
		T norm = T( 1.0 / ((double)n * std::pow((double)h0,dim)) );
		T h2 = h0*h0;

#ifdef MEANSHIFT_ENABLE_OMP
		#pragma omp parallel for
#endif
	    for( int i=0; (unsigned)i < n; ++i )
	    {
#ifdef MEANSHIFT_ENABLE_OMP
            // When working in parallel we require a private sqdist buffer (local to each thread)
            T* sqdist = new T[n];
#endif
			const T* xi = &X[i*dim];
			dist2( xi, X, n,dim, sqdist );
			
			density[i] = T(0.0);
			for( unsigned j=0; j < n; ++j )
				density[i] += exp( -sqdist[j] / h2 );
			density[i] *= norm;

#ifdef MEANSHIFT_ENABLE_OMP
            delete [] sqdist;
#endif
		}
		
#ifndef MEANSHIFT_ENABLE_OMP
		delete [] sqdist;
#endif
	}
	
protected:
    //--------
    // NOTE:
    // For the following helper functions we silently assume that output matrices are always pre-allocated!

    /// Compute h_i^-(d+1) normalization.
    static inline void compute_adaptive_normalization( const T* bandwidth, unsigned n, unsigned dim, T* c )
    {
        for( unsigned i=0; i < n; ++i )
            c[i] = T( 1.0 / std::pow( (double)bandwidth[i], (double)(dim+2) ) );
    }

    static inline void multiply_pointwise( const T* a, const T* b, unsigned n, T* ab )
    {
        const T *a_=a, *b_=b;
              T *ab_=ab;
        for( unsigned i=0; i < n; ++i, ++a_,++b_,++ab_ )
            (*ab_) = (*a_)*(*b_);
    }

	/// Compute squared distances from point y to each point in data matrix X.
	static inline void dist2( const T* y_, const T* X, unsigned n, unsigned dim, T* distances )
    {
	    const T *x  = X;
	          T *di = distances;
	    for( unsigned i=0; i < n; ++i, ++di )
	    {
		    // Distance x_i to y
		    const T *y = y_;
		    (*di) = T(0.0);
		    for( unsigned d=0; d < dim; ++d, ++y, ++x )
		    {
			    // Squared Euclidean distance
                T diff = (*y) - (*x);
			    (*di) += diff*diff;
		    }
	    }
    }
	
    /// Compute weights with Gaussian normal kernel based on adaptive bandwidth per point.
	static inline void normal_kernel( const T* squared_distances, unsigned n, const T* bandwidth, T* weights )
    {	
        using std::exp;

	    const T* h = bandwidth;
	    const T* d = squared_distances;
	          T* w = weights;
	    for( unsigned i=0; i < n; ++i, ++w, ++d, ++h )
	    {
		    T h2 = (*h)*(*h);
		    (*w) = exp( - (*d) / h2 );
	    }
    }
	
    /// Compute mean shifted point y based on kernel weights.
    /// Returns false in case of zero weight vector.
	static inline bool eval_meanshift( const T* X, const T* weights, unsigned n, unsigned dim, T* y_ )
    {
	    T denom = T(0.0);

	    const T *w = weights;	
	    for( unsigned i=0; i < n; ++i, ++w )
	        denom += (*w);

        // Avoid div0
        if( denom <= 1e-14 )
            return false;
	
        // For T=[float|double] it is safe to do memset 0 to set to zero
        memset( (void*)y_, 0x0, dim*sizeof(T) );

	    const T *x = X;
                 w = weights;
	    for( unsigned i=0; i < n; ++i, ++w )
	    {
		    T *y = y_;
		    for( unsigned d=0; d < dim; ++d, ++x, ++y )
			    (*y) += (*x) * (*w);
	    }
	
        {
            T *y=y_;
	        for( unsigned d=0; d < dim; ++d, ++y )
		        (*y) /= denom;
        }

        return true;
    }

private:
	Options m_opts;
    int* m_stat_iters;
};
