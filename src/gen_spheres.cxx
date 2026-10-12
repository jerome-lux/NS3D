/*
 * gen_spheres.cxx
 *
 * Test-image generator for the NS3D solver.
 *
 * Produces a binary 3D image of a regular lattice of NON-overlapping
 * spheres. The image is written as a VTK ASCII/XML image (.vti) with
 * 8-bit unsigned-char scalars:
 *
 *   0 = fluid (the inter-sphere matrix)
 *   1 = solid (the spheres)
 *
 * The lattice spacing is chosen so that 2*radius < spacing, which
 * guarantees that no two spheres overlap and that no sphere touches the
 * domain boundary. This gives a well-connected, periodic-looking porous
 * matrix that is a clean, reproducible benchmark for the Darcy
 * permeability solver.
 *
 * Usage:
 *   gen_spheres [output.vti]
 *
 * If no output path is given, "spheres_128.vti" is written in the
 * current directory.
 */

// std headers
#include <cstdio>
#include <cstdlib>
#include <string>

// Blitz++ for 3D arrays
#include <blitz/array.h>

// project headers
#include "vtkio.h"

/* ------------------------------------------------------------------------ */
/* Geometry parameters                                                      */
/* ------------------------------------------------------------------------ */

/**
 * @brief Fill a N^3 binary array with a lattice of non-overlapping spheres.
 *
 * Spheres of radius @p radius (voxels) are placed on a cubic lattice whose
 * spacing is @p spacing (voxels). Centers are located at the middle of each
 * lattice cell: (spacing/2 + spacing*m, ...) with m, n, p in 0..(N/spacing-1).
 * The constraint 2*radius < spacing guarantees that adjacent spheres do not
 * overlap, and because each center is at least spacing/2 (>= radius+1, from
 * the same constraint) away from any domain face, no sphere is cut by the
 * boundary. This placement is size-independent, so it works for any N.
 *
 * @param image  [in,out] array to fill; must already be sized to (N,N,N).
 *               It is first reset to 0 (fluid) and the sphere voxels are
 *               then set to 1 (solid).
 * @param N      image size (N^3).
 * @param spacing lattice spacing between sphere centers (voxels).
 * @param radius sphere radius (voxels).
 * @param nspheres [out] number of spheres placed.
 */
static void MakeSphereLattice(blitz::Array<unsigned char, 3> &image,
                              int N, int spacing, int radius,
                              int &nspheres)
{
    image = 0;    // start with an all-fluid matrix
    nspheres = 0;

    const int nper = N / spacing;                 // spheres per axis
    const double r2 = static_cast<double>(radius) * radius;

    for (int m = 0; m < nper; ++m)
        for (int n = 0; n < nper; ++n)
            for (int p = 0; p < nper; ++p)
            {
                const int cx = spacing / 2 + spacing * m;
                const int cy = spacing / 2 + spacing * n;
                const int cz = spacing / 2 + spacing * p;

                // bounding box of the sphere, clamped to the domain
                int lo_x = cx - radius; if (lo_x < 0) lo_x = 0;
                int hi_x = cx + radius; if (hi_x > N - 1) hi_x = N - 1;
                int lo_y = cy - radius; if (lo_y < 0) lo_y = 0;
                int hi_y = cy + radius; if (hi_y > N - 1) hi_y = N - 1;
                int lo_z = cz - radius; if (lo_z < 0) lo_z = 0;
                int hi_z = cz + radius; if (hi_z > N - 1) hi_z = N - 1;

                for (int i = lo_x; i <= hi_x; ++i)
                    for (int j = lo_y; j <= hi_y; ++j)
                        for (int k = lo_z; k <= hi_z; ++k)
                        {
                            const double d2 =
                                static_cast<double>(i - cx) * (i - cx)
                              + static_cast<double>(j - cy) * (j - cy)
                              + static_cast<double>(k - cz) * (k - cz);
                            if (d2 <= r2)
                                image(i, j, k) = 1;    // solid
                        }
                ++nspheres;
            }
}

/* ------------------------------------------------------------------------ */
/* Main                                                                     */
/* ------------------------------------------------------------------------ */

/**
 * @brief Program entry point: generate and write the test image.
 *
 * Usage:
 *   gen_spheres [output.vti] [N] [spacing] [radius]
 *
 * All arguments are optional. Defaults produce the 128^3 benchmark image.
 * Examples:
 *   gen_spheres data/spheres_128.vti              # 128^3, radius 5, spacing 12
 *   gen_spheres data/spheres_64.vti  64 8 3       #  64^3, radius 3, spacing 8
 *
 * @param argc argument count.
 * @param argv argv[1]=output, argv[2]=N, argv[3]=spacing, argv[4]=radius.
 * @return 0 on success, 1 on a parameter error.
 */
int main(int argc, char *argv[])
{
    // Parse the (optional) command-line arguments.
    std::string out = (argc > 1) ? std::string(argv[1])
                                 : std::string("spheres_128.vti");
    int N       = (argc > 2) ? std::atoi(argv[2]) : 128;
    int spacing = (argc > 3) ? std::atoi(argv[3]) : 12;
    int radius  = (argc > 4) ? std::atoi(argv[4]) : 5;

    if (N <= 0 || spacing <= 0 || radius <= 0)
    {
        std::fprintf(stderr, "error: N, spacing and radius must all be > 0\n");
        return 1;
    }
    if (2 * radius >= spacing)
    {
        std::fprintf(stderr,
                     "error: need 2*radius (%d) < spacing (%d) to avoid overlap\n",
                     2 * radius, spacing);
        return 1;
    }

    blitz::Array<unsigned char, 3> image(N, N, N);
    int nspheres = 0;
    MakeSphereLattice(image, N, spacing, radius, nspheres);

    // statistics
    const long total = static_cast<long>(N) * N * N;
    long solid = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                if (image(i, j, k) == 1)
                    ++solid;

    const double solid_frac = static_cast<double>(solid) / static_cast<double>(total);

    std::printf("Generated sphere lattice image\n");
    std::printf("  size        : %d^3 (%ld voxels)\n", N, total);
    std::printf("  spheres     : %d (radius %d, spacing %d)\n", nspheres, radius, spacing);
    std::printf("  solid       : %ld voxels (%.4f)\n", solid, solid_frac);
    std::printf("  porosity    : %.4f\n", 1.0 - solid_frac);

    writeBinaryImage(out, image);

    std::printf("Done: %s\n", out.c_str());
    return 0;
}
