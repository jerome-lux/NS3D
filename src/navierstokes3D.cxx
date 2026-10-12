/*
 * navierstokes3D.cxx
 *
 * Copyright 2016 Jerome Lux <jlux@eduroam2058.univ-lr.fr>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 *
 * Usage:
 *   NS3D <input-file>
 *
 * Use OMP_NUM_THREADS=number_of_real_cores ./NS3D for best performance.
 */

// std headers
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

// project headers
#include "IDEAL.h"
#include "SIMPLERR.h"
#include "SIMPLEC.h"
#include "SIMPLER.h"
#include "NSBase.h"

/* ------------------------------------------------------------------------ */
/* Input parsing helpers                                                    */
/* ------------------------------------------------------------------------ */

/**
 * Strip leading/trailing whitespace (spaces, tabs, CR/LF) from a string.
 *
 * @param s input string.
 * @return trimmed copy of s (empty string if s is all whitespace).
 */
static std::string trim(const std::string &s)
{
    const char *ws = " \t\r\n";
    std::size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos)
        return std::string();
    std::size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

/**
 * Convert an ASCII string to lowercase.
 *
 * @param s input string.
 * @return lowercase copy of s.
 */
static std::string tolower(const std::string &s)
{
    std::string r = s;
    for (char &c : r)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    return r;
}

/**
 * Convert an ASCII string to uppercase.
 *
 * @param s input string.
 * @return uppercase copy of s.
 */
static std::string toupper(const std::string &s)
{
    std::string r = s;
    for (char &c : r)
        if (c >= 'a' && c <= 'z')
            c = static_cast<char>(c - 'a' + 'A');
    return r;
}

/**
 * Parse a double strictly: the whole string must be a valid number.
 *
 * @param s   text to parse.
 * @param out parsed value (set only on success).
 * @return true on success, false otherwise.
 */
static bool parse_double(const std::string &s, double &out)
{
    if (s.empty())
        return false;
    std::size_t pos = 0;
    try {
        out = std::stod(s, &pos);
    } catch (...) {
        return false;
    }
    return pos == s.size();
}

/**
 * Parse an integer strictly: the whole string must be a valid integer.
 *
 * @param s   text to parse.
 * @param out parsed value (set only on success).
 * @return true on success, false otherwise.
 */
static bool parse_int(const std::string &s, int &out)
{
    if (s.empty())
        return false;
    std::size_t pos = 0;
    try {
        out = std::stoi(s, &pos);
    } catch (...) {
        return false;
    }
    return pos == s.size();
}

/**
 * Remove an inline comment from a value.
 *
 * A '#' starts a comment only if it is the first non-empty character of
 * the value or is preceded by whitespace, so file names containing '#'
 * without a preceding space stay intact.
 *
 * @param v value string.
 * @return value with the inline comment removed (and trimmed).
 */
static std::string strip_comment(const std::string &v)
{
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        if (v[i] == '#' && (i == 0 || std::isspace(static_cast<unsigned char>(v[i - 1]))))
            return trim(v.substr(0, i));
    }
    return trim(v);
}

/* ------------------------------------------------------------------------ */
/* Parameter container                                                      */
/* ------------------------------------------------------------------------ */

/**
 * @brief Container for every simulation parameter.
 *
 * Every field carries a default value; ReadInput() overrides the fields
 * that are present in the input file and reports the effective values.
 */
struct InputParams
{
    // files
    std::string image;
    std::string pressurefield = "none";
    std::string vxfield = "none";
    std::string vyfield = "none";
    std::string vzfield = "none";

    // physics
    double dx = 1.0;
    double dy = -1.0;    // negative value: use dx instead
    double dz = -1.0;    // negative value: use dx instead
    double mu = 1e-3;
    double rho = 1e3;
    double p0 = 1e5;
    double gradp = 1e2;

    // numerics
    double alpha = 0.8;
    double beta = 1.0;
    double vsolvertol = 1e-5;
    double psolvertol = 1e-6;
    double pcsolvertol = 1e-3;
    double simpletol = 1e-4;
    double deltaRMR = 1e-4;
    int maxitsolver = 1000;
    int maxitSIMPLE = 500;
    int N1 = 2;
    int N2 = 2;

    // geometry
    int porevalue = 0;
    int direction = 1;   // 1:x, 2:y, 3:z
    int border = 0;

    // algorithm
    std::string method = "SIMPLEC";
};

/* ------------------------------------------------------------------------ */
/* Input file parsing                                                       */
/* ------------------------------------------------------------------------ */

/**
 * @brief Print a short usage message and the list of accepted keys.
 */
static void PrintUsage()
{
    printf("Usage: NS3D <input-file>   (or 'NS3D -h' for this help)\n");
    printf("\nInput file format: one 'key = value' per line.\n");
    printf("Keys are case-insensitive, blank lines and lines starting\n");
    printf("with '#' are ignored, and an inline '# ...' comment is\n");
    printf("allowed after a value. Unknown keys are reported as a warning.\n");
    printf("\nAccepted keys:\n");
    printf("  input          image file (binary 3D image, .vti or .vtk)   [required]\n");
    printf("  pressurefield  initial pressure field file ('none' to skip)\n");
    printf("  vxfield        initial x-velocity field file ('none' to skip)\n");
    printf("  vyfield        initial y-velocity field file ('none' to skip)\n");
    printf("  vzfield        initial z-velocity field file ('none' to skip)\n");
    printf("  dx, dy, dz     grid spacing (m); dy/dz < 0 means 'same as dx'\n");
    printf("  mu             dynamic viscosity (Pa.s)\n");
    printf("  rho            density (kg/m^3)\n");
    printf("  p0             inlet pressure (Pa)\n");
    printf("  gradp          imposed pressure gradient (Pa/m)\n");
    printf("  alpha          velocity relaxation parameter\n");
    printf("  beta           pressure relaxation parameter\n");
    printf("  vsolvertol     velocity solver tolerance\n");
    printf("  psolvertol     pressure solver tolerance\n");
    printf("  pcsolvertol    pressure correction solver tolerance\n");
    printf("  simpletol      outer-loop convergence threshold (relative RMR)\n");
    printf("  deltaRMR       oscillation detection threshold\n");
    printf("  maxitsolver    max iterations of the linear solvers\n");
    printf("  maxitSIMPLE    max outer iterations\n");
    printf("  N1             inner pressure iterations (SIMPLER/SIMPLERR/IDEAL)\n");
    printf("  N2             inner final iterations (IDEAL)\n");
    printf("  porevalue      voxel value of the fluid (0 or 1)\n");
    printf("  direction      pressure gradient direction: 1=x, 2=y, 3=z\n");
    printf("  border         solid border thickness added along x (voxels)\n");
    printf("  method         algorithm: SIMPLEC, SIMPLER, SIMPLERR or IDEAL\n");
}

/**
 * @brief Parse a 'key = value' input file into an InputParams structure.
 *
 * Behaviour:
 *  - blank lines and lines starting with '#' are skipped;
 *  - keys are matched case-insensitively; whitespace around the key and
 *    the value is ignored; an inline '# ...' comment after the value
 *    is accepted;
 *  - an unknown key produces a warning (with the line number) and the
    line is ignored;
 *  - a malformed value (e.g. a non-numeric value for 'mu') or a missing
 *    required key ('input') is a fatal error;
 *  - after parsing, defaults are applied (dy/dz < 0 -> dx) and the
 *    effective parameter set is printed.
 *
 * @param inifile  opened input file stream.
 * @param filename file name (used in error messages).
 * @param p        parameters to fill.
 * @return true if the file was parsed without fatal error, false otherwise.
 */
static bool ReadInput(std::ifstream &inifile, const std::string &filename, InputParams &p)
{
    bool has_image = false;

    std::string line;
    std::size_t lineno = 0;

    while (std::getline(inifile, line))
    {
        ++lineno;
        line = trim(line);

        if (line.empty() || line[0] == '#')
            continue;

        std::size_t eq = line.find('=');
        if (eq == std::string::npos)
        {
            std::cerr << "warning: " << filename << ":" << lineno
                      << " - line ignored (no '=' found)\n";
            continue;
        }

        std::string key = tolower(trim(line.substr(0, eq)));
        std::string value = strip_comment(line.substr(eq + 1));

        // report a fatal error for a malformed value
        auto bad_value = [&](const char *what) -> bool {
            std::cerr << "error: " << filename << ":" << lineno
                      << " - key '" << key << "' expects a " << what
                      << ", got '" << value << "'\n";
            return false;
        };

        /* ------------------------- files ------------------------- */
        if (key == "input") {
            p.image = value;
            has_image = true;
        } else if (key == "pressurefield") {
            p.pressurefield = value;
        } else if (key == "vxfield") {
            p.vxfield = value;
        } else if (key == "vyfield") {
            p.vyfield = value;
        } else if (key == "vzfield") {
            p.vzfield = value;

        /* ------------------------- physics ------------------------- */
        } else if (key == "dx") {
            if (!parse_double(value, p.dx)) return bad_value("number");
        } else if (key == "dy") {
            if (!parse_double(value, p.dy)) return bad_value("number");
        } else if (key == "dz") {
            if (!parse_double(value, p.dz)) return bad_value("number");
        } else if (key == "mu") {
            if (!parse_double(value, p.mu)) return bad_value("number");
        } else if (key == "rho") {
            if (!parse_double(value, p.rho)) return bad_value("number");
        } else if (key == "p0") {
            if (!parse_double(value, p.p0)) return bad_value("number");
        } else if (key == "gradp") {
            if (!parse_double(value, p.gradp)) return bad_value("number");

        /* ------------------------ numerics ------------------------- */
        } else if (key == "alpha") {
            if (!parse_double(value, p.alpha)) return bad_value("number");
        } else if (key == "beta") {
            if (!parse_double(value, p.beta)) return bad_value("number");
        } else if (key == "vsolvertol") {
            if (!parse_double(value, p.vsolvertol)) return bad_value("number");
        } else if (key == "psolvertol") {
            if (!parse_double(value, p.psolvertol)) return bad_value("number");
        } else if (key == "pcsolvertol") {
            if (!parse_double(value, p.pcsolvertol)) return bad_value("number");
        } else if (key == "simpletol") {
            if (!parse_double(value, p.simpletol)) return bad_value("number");
        } else if (key == "deltaRMR") {
            if (!parse_double(value, p.deltaRMR)) return bad_value("number");
        } else if (key == "maxitsolver") {
            if (!parse_int(value, p.maxitsolver)) return bad_value("integer");
        } else if (key == "maxitsimple") {
            if (!parse_int(value, p.maxitSIMPLE)) return bad_value("integer");
        } else if (key == "n1") {
            if (!parse_int(value, p.N1)) return bad_value("integer");
        } else if (key == "n2") {
            if (!parse_int(value, p.N2)) return bad_value("integer");

        /* ------------------------- geometry ------------------------ */
        } else if (key == "porevalue") {
            if (!parse_int(value, p.porevalue)) return bad_value("integer");
        } else if (key == "direction") {
            if (!parse_int(value, p.direction)) return bad_value("integer");
        } else if (key == "border") {
            if (!parse_int(value, p.border)) return bad_value("integer");

        /* ------------------------- algorithm ----------------------- */
        } else if (key == "method") {
            p.method = value;
        } else {
            std::cerr << "warning: " << filename << ":" << lineno
                      << " - unknown key '" << key << "' (ignored)\n";
        }
    }

    /* ----------------------- validation --------------------------- */

    if (!has_image)
    {
        std::cerr << "error: " << filename << " - required key 'input' (image file) is missing\n";
        return false;
    }
    if (p.image.empty())
    {
        std::cerr << "error: " << filename << " - key 'input' has an empty value\n";
        return false;
    }
    if (p.dx <= 0.0)
    {
        std::cerr << "error: " << filename << " - 'dx' must be > 0\n";
        return false;
    }
    if (p.mu <= 0.0)
    {
        std::cerr << "error: " << filename << " - 'mu' must be > 0\n";
        return false;
    }
    if (p.rho <= 0.0)
    {
        std::cerr << "error: " << filename << " - 'rho' must be > 0\n";
        return false;
    }
    if (p.maxitsolver <= 0)
    {
        std::cerr << "error: " << filename << " - 'maxitsolver' must be > 0\n";
        return false;
    }
    if (p.maxitSIMPLE <= 0)
    {
        std::cerr << "error: " << filename << " - 'maxitSIMPLE' must be > 0\n";
        return false;
    }
    if (p.porevalue != 0 && p.porevalue != 1)
    {
        std::cerr << "warning: " << filename << " - 'porevalue' should be 0 or 1, "
                  << "treating '" << p.porevalue << "' as 0\n";
        p.porevalue = 0;
    }
    if (p.direction < 1 || p.direction > 3)
    {
        std::cerr << "error: " << filename << " - 'direction' must be 1 (x), 2 (y) or 3 (z)\n";
        return false;
    }

    // 'method' is matched case-insensitively; the original code silently
    // fell back to SIMPLEC for any unknown value, which is a foot-gun.
    std::string m = tolower(p.method);
    if (m != "simplec" && m != "simpler" && m != "simplerr" && m != "ideal")
    {
        std::cerr << "error: " << filename << " - unknown method '" << p.method
                  << "'. Valid values: SIMPLEC, SIMPLER, SIMPLERR, IDEAL\n";
        return false;
    }
    p.method = toupper(p.method);

    // defaults for dy / dz
    if (p.dy <= 0.0)
        p.dy = p.dx;
    if (p.dz <= 0.0)
        p.dz = p.dx;

    /* --------------------- parameter summary ---------------------- */
    printf("=== Input parameters ===\n");
    printf("method         : %s\n", p.method.c_str());
    printf("image          : %s\n", p.image.c_str());
    printf("dx, dy, dz (m) : %g, %g, %g\n", p.dx, p.dy, p.dz);
    printf("mu (Pa.s)      : %g\n", p.mu);
    printf("rho (kg/m^3)   : %g\n", p.rho);
    printf("p0 (Pa)        : %g\n", p.p0);
    printf("gradp (Pa/m)   : %g\n", p.gradp);
    printf("alpha, beta    : %g, %g\n", p.alpha, p.beta);
    printf("simpletol      : %g\n", p.simpletol);
    printf("maxitSIMPLE    : %d\n", p.maxitSIMPLE);
    printf("maxitsolver    : %d\n", p.maxitsolver);
    printf("porevalue      : %d\n", p.porevalue);
    printf("direction      : %d\n", p.direction);
    printf("border         : %d\n", p.border);
    printf("======================\n");

    return true;
}

/* ------------------------------------------------------------------------ */
/* Main                                                                     */
/* ------------------------------------------------------------------------ */

/**
 * @brief Program entry point.
 *
 * Reads the input file, builds the NavierStokesBase instance (geometry,
 * index arrays, initial fields) and dispatches to the requested algorithm
 * (SIMPLEC, SIMPLER, SIMPLERR or IDEAL).
 *
 * @param argc argument count.
 * @param argv argument values.
 * @return 0 on success, 1 on input error.
 */
int main(int argc, char *argv[])
{
#ifdef EIGEN_HAS_OPENMP
    printf("Eigen has openmp support\n");
#else
    printf("!! Eigen has no openmp support !!\n");
#endif

    Eigen::initParallel();

    if (argc < 2)
    {
        PrintUsage();
        return 1;
    }

    std::string inifilename = argv[1];

    if (inifilename == "-h" || inifilename == "--help")
    {
        PrintUsage();
        return 0;
    }

    std::ifstream inifile(inifilename.c_str(), std::ios::in);
    if (!inifile.is_open())
    {
        std::cerr << "error: cannot open input file '" << inifilename << "'\n";
        return 1;
    }

    // read the parameters
    InputParams p;
    if (!ReadInput(inifile, inifilename, p))
        return 1;

    NavierStokesBase SolverBase(p.image, p.pressurefield, p.vxfield, p.vyfield, p.vzfield,
                                p.dx, p.dy, p.dz,
                                p.vsolvertol, p.psolvertol, p.pcsolvertol, p.maxitsolver,
                                p.alpha, p.beta, p.p0, p.gradp, p.mu, p.rho,
                                p.porevalue, p.direction, p.deltaRMR, p.border);

    // create the requested solver and run it
    if (p.method == "IDEAL")
    {
        IDEALSOLVER solver(p.simpletol, p.maxitSIMPLE, p.N1, p.N2);
        solver.Solve(SolverBase);
    }
    else if (p.method == "SIMPLER")
    {
        SIMPLERSOLVER solver(p.simpletol, p.maxitSIMPLE, p.N1);
        solver.Solve(SolverBase);
    }
    else if (p.method == "SIMPLERR")
    {
        SIMPLERRSOLVER solver(p.simpletol, p.maxitSIMPLE, p.N1);
        solver.Solve(SolverBase);
    }
    else
    {
        SIMPLECSOLVER solver(p.simpletol, p.maxitSIMPLE);
        solver.Solve(SolverBase);
    }

    return 0;
}
