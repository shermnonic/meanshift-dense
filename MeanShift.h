#pragma once
/** 
  \file    MeanShift.h
  \brief   Header-only dense mean-shift algorithm for small point clouds.
  \version 1.1
  \author  Max Hermann (mnemonic@386dx25.de)
  \date    2015-2019
  \copyright MIT License

  Revision history:
  - v1.1 2019/04 
    - More convenient computeMeanShiftInPlace() function with automatic bandwidth estimation.
    - Support for callback on each complete iteration e.g. to assemble trajectories from intermediate state.
    - Requires C++11, because of std::function.
  - v1.0 2015/11 
    - Initial version providing shift_points(), pure C++98, no internal data.
*/

#include <cstring>     // memcpy()
#include <cmath>       // exp(),pow(),sqrt()
#include <algorithm>   // max_element()
#include <functional>
#include <vector>

/**
  Mean-shift clustering with adaptive bandwidth.
  
  This is a brute force implementation w/o discretization for efficient
  processing of small datasets with n<1500 points of up to d<8 dimensions.
  The class is self-contained, i.e. has no additional dependencies beyond std,
  operates on raw data pointers and is header-only.

  The main routine is computeMeanShiftInPlace() that implements mean-shift of a 
  set of test points Y against a density estimate of another set of data points X.
  The data points X are passed into the c'tor as const pointer which is assumed 
  to be valid during the whole lifetime of the MeanShift instance. 
  The test points Y, that might be a copy of X but should not be a pointer to X, 
  is passed into computeMeanShiftInPlace().
  
  Note that the routine does not compute a clustering/segmentation but outputs only
  the mean-shifted test points that, in case the mean-shift algorithm has
  converged, will accumulate at candidate positions for maximum modes of 
  the density. Identification of stable modes (e.g. via perturbation analysis)
  and deriving a clustering from the result is up to the callee.

  \tparam T   The scalar type, either float or double.
  \tparam DIM Dimension of the data points.
*/
template<class T, int DIM>
class MeanShift
{
private:
    const T* m_data;      ///< Data points matrix X of size numPoints x DIM
    size_t   m_numPoints; ///< Number of data points (i.e. rows) in X

    unsigned m_maxIter;   ///< Max. number of mean-shift iterations
    T        m_epsilon;   ///< Epsilon, usually estimated per data matrix X
    T        m_h0;        ///< Pilot bandwidth, usually estimated per data matrix X
    std::vector<T>   m_bandwidth;  ///< Bandwidth parameter per data point in matrix X
    std::vector<int> m_stat_iters; ///< Number of iterations per test point until convergence or -1 if mean-shift failed

public:
    /** Construct a MeanShift instance to be executed on the n x DIM data matrix.

        @param[in] X         n x DIM data matrix of n points of dimension DIM (*must* stay valid during instance lifetime!)
        @param[in] numPoints number of data points in X
        @param[in] maxIter   maximum number of iterations (default value of 100 should usually be sufficient)
    */
    MeanShift( const T* X, size_t numPoints, unsigned maxIter=100 )
    : m_data(X), 
      m_numPoints(numPoints),
      m_maxIter(maxIter)
    {}

    ~MeanShift() = default;

    /** Compute mean-shift of the (m x DIM) test points matrix Y in-place.

        @param[in,out] Y      pre-allocated m x d data matrix of test points to which mean-shift is applied (in-place)
        @param[in] m          number of test points in Y
    */
    void computeMeanShiftInPlace( T* Y, unsigned m )
    {
        ensureBandwidth();
        shift_points( m_data, &m_bandwidth[0], (unsigned)m_numPoints, Y, m );
    }

    /** Compute mean-shift on the (m x DIM) test points matrix Y in-place where
        the provided callback is invoked on each complete iteration of all test points.

        The callback function arguments are (const T* Y, int iter), i.e. the current shifted test points matrix Y 
        (of size m x DIM) and the current iteration number between 0 and \a maxIter(), whereas iter=0 corresponds to 
        the initial input test points matrix and iter=maxIter() to the final state.

        @param[in,out] Y      pre-allocated m x d data matrix of test points to which mean-shift is applied (in-place)
        @param[in] m          number of test points in Y
        @param[in] callbackOnIter  is a function that is called on each complete iteration, see description.
    */
    void computeMeanShiftInPlace( T* Y, unsigned m, std::function<void(const T*,int)> callbackOnIter )
    {
        ensureBandwidth();
        shift_points( m_data, &m_bandwidth[0], (unsigned)m_numPoints, Y, m, callbackOnIter );
    }

    ///@name Mean-shift parameters 
    ///@{

    /// Return last used threshold epsilon
    inline T epsilon() const { return m_epsilon; }
    /// Return last used pilot bandwidth
    inline T h0() const { return m_h0; }
    /// Return maximum number of iterations
    inline unsigned maxIter() const { return m_maxIter; }
    ///@}

    ///@name Statistics on last mean-shift execution
    ///@{

    /// Enable or disable statistics
    void enableStats( bool b=true ) { m_opts.doStats = b; }
    /// Return array with number of iterations until convergence per point, for last succesfull mean-shift procedure
    /// for which stats were enabled. Note, that when using the computeMeanShiftInPlace() with callback, stats are
    /// always computed. Otherwise, they can be enabled via \a enableStats().
    /// For test points points where mean-shift procedure failed, a value of -1 is noted.
    std::vector<int> stats() const { return m_stat_iters; }
    ///@}

protected:
    ///@name Mean-shift computation
    ///@{

    /** Compute adaptive bandwidth non-blurring mean shift, operates in-place on the test points matrix Y.
        @param[in] X          n x d data matrix of n points of dimension d
        @param[in] bandwidth  vector with n scale parameters, one for each point
        @param[in] n          number of data points in X
        @param[in,out] Y      pre-allocated m x d data matrix of test points to which mean-shift is applied (in-place)
        @param[in] m          number of test points in Y

        Loops are split for good parallel peformance, i.e. the outer (parallel) 
        loop is over the test points and the inner loop is over the iterations.
        If you want access to intermediate positions of shifted test points, 
        see the other shift_points() variant.
    */
    void shift_points( const T* X, const T* bandwidth, unsigned n, T* Y, unsigned m
#ifdef MEANSHIFT_ENABLE_INEFFICIENT_CALLBACK
        ,std::function<void(int,int,const T*)> callback = std::function<void(int, int, const T*)>()
#endif
        )
    {
        T* Y0        = new T[m*DIM];
        T* distances = new T[n];
        T* weights   = new T[n];
        T* norm      = new T[n];

        // Stats are optional 
        if( m_opts.doStats )
            initStats(m);

        // Pre-compute normalization
        compute_adaptive_normalization( bandwidth, n, DIM, norm );
    
        // Iterate all points in Y
        memcpy( (void*)Y0, (void*)Y, m*DIM*sizeof(T) );
        
#ifdef MEANSHIFT_ENABLE_OPENMP
        #pragma omp parallel for
#endif
        for( int i=0; i < static_cast<int>(m); ++i )
        {
            T* y0i = &Y0[i*DIM];
            T* yi  = &Y [i*DIM];

#ifdef MEANSHIFT_ENABLE_INEFFICIENT_CALLBACK
            if( callback ) callback(0,i,yi);
#endif
        
            unsigned iter=0;
            bool halt=false;
            while( iter < m_maxIter && !halt)
            {
                StepState state = shift_step(distances, norm, y0i, yi, X, bandwidth, weights, n, DIM);
                switch( state )
                {
                case StepFailed:
                    // Skip iteration for this point.
                    halt = true;
                    if (m_opts.doStats) m_stat_iters[i] = -1;
                    break;

                case StepConverged:
                    halt = true;
                    if (m_opts.doStats) m_stat_iters[i] = iter+1;
                    break;

                default:
                case StepSuccess:
                    break;
                }

                iter++;

#ifdef MEANSHIFT_ENABLE_INEFFICIENT_CALLBACK
                if (callback) callback(iter, i, yi);
#endif
            }
        }
    
        delete [] norm;
        delete [] weights;
        delete [] distances;
        delete [] Y0;
    }

    /** Compute adaptive bandwidth non-blurring mean shift with callback on each complete iteration 
        (e.g. to assemble trajectories), operates in-place on the test points matrix Y.

        @param[in] callbackOnIter  is a function that is called on each complete iteration, its arguments are
                              (const T* Y, int iter) the current shifted test points matrix Y (of size m x DIM) and
                              the current iteration number between 0 and \a maxIter(), whereas iter=0 corresponds to 
                              the initial input test points matrix and iter=maxIter() to the final state.

        Loops are split to yields trajectories for each complete iteration, i.e.
        the outer loop is over iterations, the inner (parallel) loop over points.
        If you do not need the callback, see the other shift_points() variant that
        might provide better parallel performance.
    */
    void shift_points(const T* X, const T* bandwidth, unsigned n, T* Y, unsigned m,
        std::function<void(const T*,int)> callbackOnIter
        // Note that using std::function here as calback is chosen over
        // void(*callbackOnIter)(const T* Yi) because it allows stateful lambdas.
        )
    {
        T* Y0        = new T[m*DIM];
        T* norm      = new T[n];
#ifndef MEANSHIFT_ENABLE_OPENMP
        T* distances = new T[n];
        T* weights   = new T[n];
#endif
        // Stats are required to determine thread execution
        initStats(m);

        // Pre-compute normalization
        compute_adaptive_normalization( bandwidth, n, norm );

        // Iterate all points in Y
        memcpy((void*)Y0, (void*)Y, m*DIM * sizeof(T));
        for( unsigned iter = 0; iter < m_maxIter; ++iter )
        {
            // Output trajectories
            callbackOnIter(Y, static_cast<int>(iter));

#ifdef MEANSHIFT_ENABLE_OPENMP
            #pragma omp parallel for
#endif
            for( int i = 0; i < static_cast<int>(m); ++i )
            {
                bool is_halted = m_stat_iters[i]!=0;

                T* y0i = &Y0[i*DIM];
                T* yi = &Y[i*DIM];
                if( !is_halted )
                {
#ifndef MEANSHIFT_ENABLE_OPENMP
                    StepState state = shift_step(distances, norm, y0i, yi, X, bandwidth, weights, n);
#else
                    StepState state = shift_step(nullptr,   norm, y0i, yi, X, bandwidth, nullptr, n);
#endif
                        
                    switch( state )
                    {
                    case StepFailed:
                        // Skip iteration for this point.
                        m_stat_iters[i] = -1;
                        break;

                    case StepConverged:
                        m_stat_iters[i] = iter+1;
                        break;

                    default:
                    case StepSuccess:
                        break;
                    }
                }
                else
                {
                    // Just repeat last valid/converged point position
                    memcpy((void*)y0i, (void*)yi, DIM * sizeof(T));
                }
            }
        }

        // Output trajectories
        callbackOnIter(Y, static_cast<int>(m_maxIter));
    }

    /// Return values of \a shift_step()
    enum StepState
    {
        StepFailed,    /*!< Mean shift step could not be computed; trying again with increased bandwidth might help. */
        StepSuccess,   /*!< Mean shift step performed and not converged yet, continue mean shift iteration. */
        StepConverged  /*!< Mean shift step converged, step delta in last iteration was below threshold epsilon. */
    };

    /** Perform a single step of the mean shift procedure, i.e. this is the inner-most function of shift_points().
    */
    StepState shift_step(T* distances_, const T* norm, T* y0i, T* yi, const T* X, const T* bandwidth, T* weights_, unsigned n)
    {
        // If available, use global buffers, otherwise allocate locally (TODO: Replace by thread-local buffers)
        T* distances = distances_ ? distances_ : new T[n];
        T* weights   = weights_   ? weights_   : new T[n];

        // Mean shift
        dist2(y0i, X, n, distances);                     // d_i  = ||y_i - x_i||^2
        normal_kernel(distances, n, bandwidth, weights); // w'_i = exp( -d_i / (h_i*h_i) )
        multiply_pointwise(weights, norm, n, weights);   // w_i  = (1 / h_i^(d+2)) * w'_i
        if (!eval_meanshift(X, weights, n, yi))          // y_j = sum_i( x_i*w_i ) / sum_i( w_i )
        {
            // Mean-shift could not be computed!
            // This may happen in case of zero weights, i.e. test points 
            // are too far off the estimated density or bandwidth is chosen
            // way too small.

            if( !distances_ ) delete [] distances;
            if( !weights_   ) delete [] weights;

            // Do not update test point
            return StepFailed;
        }

        // Check for convergence
        bool converged = false;
        {
            T delta;
            dist2(y0i, yi, 1, &delta);
            converged = std::sqrt(delta) < m_epsilon;
        }

        // Update
        memcpy( (void*)y0i, (void*)yi, DIM*sizeof(T) );

        if( !distances_ ) delete [] distances;
        if( !weights_   ) delete [] weights;

        return converged ? StepConverged : StepSuccess;
    }
    
    ///@}

    ///@name Bandwidth computation heuristics
    ///@{

    /// Convenience function that computes bandwidth if not set.
    void ensureBandwidth()
    {
        if( m_bandwidth.size() != m_numPoints )
        {
            computeSilvermanPilotBandwidth( m_data, (unsigned)m_numPoints, m_h0, m_epsilon );
            m_bandwidth.resize( (unsigned)m_numPoints, m_h0 );
            computeVariableBandwidth( m_data, (unsigned)m_numPoints, m_h0, &m_bandwidth[0] );
        }
    }

    /** Compute pilot density estimate based on constant global bandwidth h0.
        This can be useful to derive an adaptive bandwidth per point.
        @param[out] density   pre-allocated 1 x n matrix for result density
    */
    static void computePilotDensityEstimate( const T* X, unsigned n, T h0, T* density )
    {
        T* sqdist  = new T[n];
        
        T norm = T( 1.0 / ((double)n * std::pow((double)h0,DIM)) );
        T h2 = h0*h0;
        
        for( unsigned i=0; i < static_cast<int>(n); ++i )
        {
            const T* xi = &X[i*DIM];
            dist2( xi, X, n, sqdist );
            
            density[i] = T(0.0);
            for( unsigned j=0; j < n; ++j )
                density[i] += exp( -sqdist[j] / h2 );
            density[i] *= norm;
        }
        
        delete [] sqdist;
    }

    /** Compute bandwidth parameter per point given a pilot bandwidth h0.
        Internally invokes computePilotDensityEstimate().
        \sa computeSilvermanPilotBandwidth()
    */
    static void computeVariableBandwidth( const T* X, unsigned n, T h0, T* bandwidth )
    {
        T* density = new T[n];
        computePilotDensityEstimate( X, n, h0, density );

        T sum = 0.;
        for (size_t i = 0; i < n; ++i)
            sum += std::log( density[i] );

        T lambda = std::exp(sum / n);
        for (size_t i = 0; i < n; ++i)
            bandwidth[i] = std::sqrt(lambda / density[i]) * h0;
    }

    /** Silverman heuristic to estimate pilot bandwidth and epsilon parameter.
        \sa computeVariableBandwidth()
    */
    static void computeSilvermanPilotBandwidth( const T* X, unsigned n, T& h0, T& epsilon )
    {
        // Silverman rule is based on max. stdev along any dimension
        T stdev;
        {
            T* mu  = new T[n];
            T* var = new T[n];

            memset((void*)mu, 0,sizeof(T)*n);
            memset((void*)var,0,sizeof(T)*n);

            for( unsigned i=0; i < n; ++i )
                for( unsigned d=0; d < DIM; ++d )
                    mu[d] += X[i*DIM+d];
            for( unsigned d=0; d < DIM; ++d )
                mu[d] /= n;

            for( unsigned i=0; i < n; ++i )
                for( unsigned d=0; d < DIM; ++d )
                    var[d] += square(X[i*DIM+d] - mu[d]);
            for( unsigned d=0; d < DIM; ++d )
                var[d] /= (n - 1);

            stdev = std::sqrt(*std::max_element(var,var+n));

            delete [] mu;
            delete [] var;
        }

        // Values 1.5 and 0.03 are heuristic
        h0 = T(1.5) * stdev * std::pow(T(n),T(-0.2));
        epsilon = T(0.03) * h0;
    }
    ///@}
    
    void initStats( unsigned m )
    {
        m_stat_iters.resize(m);
        memset((void*)m_stat_iters.data(),0,sizeof(int)*m);
    }

    ///@name Static utility functions
    /// \note The following utility functions silently assume that any output matrices are pre-allocated.
    ///@{

    static inline T square( T x ) { return x*x; }

    /// Compute h_i^-(d+1) normalization.
    static inline void compute_adaptive_normalization( const T* bandwidth, unsigned n, T* c )
    {
        for( unsigned i=0; i < n; ++i )
            c[i] = T( 1.0 / std::pow( (double)bandwidth[i], (double)(DIM+2) ) );
    }

    static inline void multiply_pointwise( const T* a, const T* b, unsigned n, T* ab )
    {
        const T *a_=a, *b_=b;
              T *ab_=ab;
        for( unsigned i=0; i < n; ++i, ++a_,++b_,++ab_ )
            (*ab_) = (*a_)*(*b_);
    }

    /// Compute squared distances from point y to each point in data matrix X.
    static inline void dist2( const T* y_, const T* X, unsigned n, T* distances )
    {
        const T *x  = X;
              T *di = distances;
        for( unsigned i=0; i < n; ++i, ++di )
        {
            // Distance x_i to y
            const T *y = y_;
            (*di) = T(0.0);
            for( unsigned d=0; d < DIM; ++d, ++y, ++x )
            {
                // Squared Euclidean distance
                T diff = (*y) - (*x);
                (*di) += square(diff);
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
            T h2 = square(*h);
            (*w) = exp( - (*d) / h2 );
        }
    }
    
    /// Compute mean shifted point y based on kernel weights.
    /// Returns false in case of zero weight vector.
    static inline bool eval_meanshift( const T* X, const T* weights, unsigned n, T* y_ )
    {
        T denom = T(0.0);

        const T *w = weights;
        for( unsigned i=0; i < n; ++i, ++w )
            denom += (*w);

        // Avoid div0
        if( denom <= 1e-14 )
            return false;
    
        // For T=[float|double] it is safe to do memset 0 to set to zero
        memset( (void*)y_, 0x0, DIM*sizeof(T) );

        const T *x = X;
                 w = weights;
        for( unsigned i=0; i < n; ++i, ++w )
        {
            T *y = y_;
            for( unsigned d=0; d < DIM; ++d, ++x, ++y )
                (*y) += (*x) * (*w);
        }
    
        {
            T *y=y_;
            for( unsigned d=0; d < DIM; ++d, ++y )
                (*y) /= denom;
        }

        return true;
    }    
    /// @}
};
