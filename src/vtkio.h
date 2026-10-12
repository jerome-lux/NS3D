/*
 * vtkio.h
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
 */

#ifndef VTKIO_H
#define VTKIO_H

// VTK
#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkImageData.h>
#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPoints.h>
#include <vtkPointData.h>
#include <vtkImageCast.h>
#include <vtkImageExport.h>

// Blitz++ for 3D arrays
#include <blitz/array.h>

// std headers
#include <iostream>
#include <string>

/**
 * @brief Read a scalar VTK image (.vti or legacy .vtk) into a 3D array.
 *
 * The file must contain double (64-bit) point scalars; the reader
 * transfers the data as-is. The copy to the Blitz++ array is
 * parallelized with OpenMP when available.
 *
 * @param inputFilename input file (.vti or .vtk).
 * @param image         [out] array filled with the scalar values.
 */
void readvtkfile_double(std::string &inputFilename, blitz::Array<double, 3> &image);

/**
 * @brief Read a binary VTK image (.vti or legacy .vtk) and binarize it.
 *
 * Voxel values are mapped to 0/1 according to @p porevalue: if
 * porevalue == 1 the fluid voxels become 1 and the solid ones 0,
 * otherwise the fluid voxels become 0 and the solid ones 1.
 *
 * @param inputFilename input file (.vti or .vtk), 8-bit point scalars.
 * @param porevalue     voxel value that represents the fluid (0 or 1).
 * @param image         [out] binarized array (0 = fluid, 1 = solid).
 */
void readvtkfile_uint8(std::string &inputFilename, int & porevalue, blitz::Array<unsigned char, 3> &image);

/**
 * @brief Write a 3D scalar array to a .vti file.
 *
 * @param filename output file name.
 * @param data     array to write.
 */
void writeScalarImage(std::string & filename, blitz::Array<double, 3> & data);

/**
 * @brief Write three 3D scalar arrays as a vector image to a .vti file.
 *
 * @param filename output file name.
 * @param u        first component.
 * @param v        second component.
 * @param w        third component.
 */
void writeVectorImage(std::string & filename, blitz::Array<double, 3> & u, blitz::Array<double, 3> & v, blitz::Array<double, 3> & w);

/**
 * @brief Write a binary 3D image (0/1) to a .vti file as 8-bit data.
 *
 * The array is stored with the VTK_UNSIGNED_CHAR scalar type. This is the
 * format expected by readvtkfile_uint8(), which reads the scalar buffer by
 * casting it to a uint8_t pointer; a file written with a wider scalar type
 * (e.g. VTK_DOUBLE) would be misread by that function.
 *
 * @param filename output file name.
 * @param data     binary array (any value; 0 = fluid, non-0 = solid).
 */
void writeBinaryImage(std::string & filename, blitz::Array<unsigned char, 3> & data);

#endif
