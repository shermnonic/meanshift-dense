/**\file meanshift-emblem.h
  \brief Demonstrate MeanShift and generate a SVG of trajectories.
 \author Max Hermann (mnemonic@386dx25.de) */
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>

#define MEANSHIFT_ENABLE_OPENMP
#include "MeanShift.h"

// Draw 2D point cloud from a mixture of Gaussians
template<typename T, typename RNG>
std::vector<T> generatePointCloud(RNG& rng, size_t numPoints = 1000, size_t k = 5)
{
    struct Gaussian2D {
        T mu_x, mu_y, sigma;
    };

    std::uniform_real_distribution<T> rand_pos(T(-1), T(1));
    std::uniform_real_distribution<T> rand_sigma(T(0.05),T(0.2));

    std::vector<std::normal_distribution<T>> mogx, mogy;
    for (size_t i = 0; i < k; ++i)
    {
        Gaussian2D g = { rand_pos(rng),rand_pos(rng),rand_sigma(rng) };
        mogx.push_back(std::normal_distribution<T>(g.mu_x,g.sigma));
        mogy.push_back(std::normal_distribution<T>(g.mu_y,g.sigma));

        std::cout << "Gaussian " << i << " at (" << g.mu_x << "," << g.mu_y << ") with sigma=" << g.sigma << std::endl;
    }

    std::vector<T> data;
    data.reserve(numPoints * 2);
    for (size_t j = 0; j < k; ++j)
    {
        auto Gx = mogx.at(j);
        auto Gy = mogy.at(j);
        for (size_t i = 0; i < numPoints / k; ++i)
        {
            T x = Gx(rng),
              y = Gy(rng);

            if (x*x + y*y < T(1))
            {
                data.push_back(x);
                data.push_back(y);
            }
        }
    }

    for (size_t i = 0; i < numPoints/3; ++i)
    {
        T x = rand_pos(rng),
          y = rand_pos(rng);
        if (x*x + y*y < T(1))
        {
            data.push_back(x);
            data.push_back(y);
        }
    }

    return data;
}

// -- SVG output functions

const double svgScale = 400.0;
const double svgOfs = 1.0;

std::string svg_start()
{
    return
    "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
    "<!-- www.386dx25.de -->\n"
    "<svg xmlns = \"http://www.w3.org/2000/svg\"\n"
    "     width=\"841mm\"\n"
    "     height=\"1189mm\"\n"
    "     viewBox=\"0 0 841 1189\">\n";
}
std::string svg_end  () { return "</svg>";  }


std::string svg_points(const std::vector<double> X, size_t numPoints)
{
    assert(X.size() == (numPoints * 2));

    std::stringstream svg;
    svg << "<g\n"
           "   id=\"points\"\n"
           "   transform=\"translate(21,195)\"\n"
           "   fill=\"gray\">\n";
    for (size_t i = 0; i < numPoints; ++i)
    {
        const double* pt = &X[i * 2];
        svg << "<circle r=\"2pt\" cx=\"" << (pt[0]+svgOfs)*svgScale << "\" cy=\"" << (pt[1]+svgOfs)*svgScale << "\"/>\n";
    }
    svg << "</g>\n";
    return svg.str();
}

std::string svg_title( std::string title )
{
    std::stringstream svg;
    svg << "<text x=\"420\" y=\"50\" style=\"text-anchor: middle\">"
        << title
        << "</text>\n";
    return svg.str();
}

std::string svg_trajectories(const std::vector<double> T, size_t numPoints, unsigned numIters)
{
    const bool smooth = true;

    std::vector<double> pts; pts.reserve(numIters*2);
    assert(T.size() == (numPoints*(numIters+1) * 2));

    std::stringstream svg;
    svg << "<g\n"
           "   id=\"trajectories\"\n"
           "   transform=\"translate(21,195)\"\n"
           "   stroke=\"black\" fill=\"none\" stroke-width=\"0.353\">\n";
    for (size_t i = 0; i < numPoints; ++i)
    {
        pts.clear();
        for (unsigned iter = 0; iter < numIters+1; ++iter)
        {
            const size_t num_values = numPoints * 2;
            const double* pt = &T[num_values*iter + i * 2];

            if (iter > 0)
            {
                const double* pt_prev = &T[num_values*(iter-1) + i * 2];
                if (pt_prev[0] == pt[0] && pt_prev[1] == pt[1])
                {
                    break;
                }
            }

            pts.push_back(pt[0]);
            pts.push_back(pt[1]);
        }
        
        if( pts.size()/2 > 2 )
        {
            std::stringstream svg_path;
            svg_path << "<path d=\"M ";
            
            if( !smooth )
            {
                for( int i=0; i < pts.size(); i+=2 )
                    svg_path << (pts[i]+svgOfs)*svgScale << " " << (pts[i+1]+svgOfs)*svgScale << " ";

            }
            else if( smooth )
            {
                // see https://stackoverflow.com/questions/1257168/how-do-i-create-a-bezier-curve-to-represent-a-smoothed-polyline#1565912
                // and http://www.ibiblio.org/e-notes/Splines/Cardinal.htm

                // gradient
                std::vector<double> grad( pts.size() );
                {
                    grad[0] = (pts[2] - pts[0])/2;
                    grad[1] = (pts[3] - pts[1])/2;

                    for( int i=2; i < pts.size()-2; i+=2 )
                    {
                        const double* p0 = &pts[i-2];
                        const double* p2 = &pts[i+2];

                        grad[i]   = (p2[0] - p0[0])/2;
                        grad[i+1] = (p2[1] - p0[1])/2;
                    }

                    size_t n = pts.size();
                    const double* pn1 = &pts[n-4];
                    const double* pn2 = &pts[n-2];
                    grad[n-2] = (pn2[0] - pn1[0])/2;
                    grad[n-1] = (pn2[1] - pn1[1])/2;
                }

                // join cubic bezier splines
                double p0x = pts[0];
                double p0y = pts[1];
                svg_path << (p0x+svgOfs)*svgScale << " " << (p0y+svgOfs)*svgScale << " "
                         << "C ";
                for( int i=1; i < pts.size()/2; ++i )
                {
                    const double* pi0 = &pts[2*(i-1)];
                    const double* pi1 = &pts[2*i];
                    const double* di0 = &grad[2*(i-1)];
                    const double* di1 = &grad[2*i];

                    double b1x = pi0[0] + di0[0]/3,
                           b1y = pi0[1] + di0[1]/3,

                           b2x = pi1[0] - di1[0]/3,
                           b2y = pi1[1] - di1[1]/3,

                           p2x = pi1[0],
                           p2y = pi1[1];

                    svg_path 
                        << (b1x+svgOfs)*svgScale << " " << (b1y+svgOfs)*svgScale << " "
                        << (b2x+svgOfs)*svgScale << " " << (b2y+svgOfs)*svgScale << " "
                        << (p2x+svgOfs)*svgScale << " " << (p2y+svgOfs)*svgScale << " ";
                }
            }
            svg_path << "\"/>\n";
            svg << svg_path.str();
        }
    }
    svg << "</g>\n";
    return svg.str();
}

// -- Utilities

class StopWatch
{
public:
    typedef std::chrono::steady_clock Clock;

    StopWatch() : m_timeStart(Clock::now()) {}

    size_t elapsed_ms() const
    {
        using std::chrono::milliseconds;
        milliseconds diff = std::chrono::duration_cast<milliseconds>(Clock::now() - m_timeStart);
        return static_cast<size_t>(diff.count());
    }

private:
    std::chrono::time_point<Clock> m_timeStart;
};


int main(int argc, char* argv[])
{
    if(argc<2)
        std::cout << "Usage: " << argv[0] << "<seedValue> <numPoints> <numGaussians>" << std::endl;

    // Arguments
    unsigned seed = std::random_device{}();
    const unsigned numIters = 100;
    size_t N = 10000;
    size_t k = 5;
    if(argc > 1)
    {
        auto user_seed = std::stoll(argv[1]);
        if(user_seed>0)
            seed = static_cast<unsigned>(user_seed);
    }
    if(argc > 2)
    {        
        N = std::stoi(argv[2]);
        k = std::stoi(argv[3]);
    }

    std::cout << "Draw " << N << " points from " << k << " Gaussians" << " and use " << seed << " as random seed." << std::endl;

    using std::to_string;
    std::string prefix = "meanshift-emblem-N" + to_string(N) + "-k" + to_string(k) + "-seed" + to_string(seed);

    std::string title = "N=" + to_string(N) + ", k=" + to_string(k) + ", s=" + to_string(seed);

    // Generate data
#ifdef _DEBUG
    std::mt19937 rng;
#else
    std::mt19937 rng(seed);
#endif
    std::vector<double> X = generatePointCloud<double>(rng,N,k);
    size_t numPoints = X.size()/2;
    {
        std::cout << "Writing points.svg...";
        std::ofstream f(prefix+".points.svg");
        f << svg_start() << svg_title(title) << svg_points(X, numPoints) << svg_end();
        std::cout << "done" << std::endl;
    }

    // Compute mean-shift w/ trajectories
    std::vector<double> Y = X;
    std::vector<double> T(Y.size() * (numIters+1));

    StopWatch tic;
    MeanShift<double,2> meanshift( X.data(), numPoints );
    meanshift.computeMeanShiftInPlace( Y.data(), (unsigned)numPoints,
        [&T,&meanshift,numPoints,numIters](const double* Yi, int iter) 
        {
            #pragma omp critical
            {
                if( iter==0 )
                {
                    std::cout << "Initial mean-shift parameters:" << std::endl;
                    std::cout << "  h0 = " << meanshift.h0() << std::endl;
                    std::cout << "  epsilon = " << meanshift.epsilon() << std::endl;
                }
                size_t num_values = numPoints * 2;
                memcpy((void*)&T[num_values*iter], (void*)Yi, sizeof(double)*num_values);
                std::cout << "Computing iteration " << iter+1 << "/" << numIters << "\r";
            }
        });
    std::cout << "Dense mean-shift of " << numPoints << " points with " << numIters << " iterations took " << tic.elapsed_ms() << "ms" << std::endl;

    // Output results
    {
        std::cout << "Writing trajectories.svg...";
        std::ofstream f(prefix+".trajectories.svg");
        f << svg_start() << svg_title(title) << svg_trajectories(T, numPoints, numIters) 
#ifdef _DEBUG
          << svg_points(X, numPoints)
#endif
          << svg_end();
        std::cout << "done" << std::endl;
    }

    return EXIT_SUCCESS;
}
