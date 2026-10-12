/*
 * vtkio.cxx
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

// std headers
#include <cstdio>

// project header
#include "vtkio.h"

/* ------------------------------------------------------------------------ */
/* Readers                                                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief Read a scalar VTK image into a Blitz++ 3D array.
 *
 * See vtkio.h for the contract. Prints the image dimensions and the
 * number of points once the file has been loaded.
 */
void readvtkfile_double(std::string &inputFilename, blitz::Array<double, 3> &image)
{
    printf("Reading image file %s ...", inputFilename.c_str());

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();

    // read the file
    if (inputFilename.size() >= 3 && inputFilename.substr(inputFilename.size() - 3) == "vti")
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(inputFilename.c_str());
        reader->Update();
        imageData->ShallowCopy(reader->GetOutput());
    }
    else if (inputFilename.size() >= 3 && inputFilename.substr(inputFilename.size() - 3) == "vtk")
    {
        vtkSmartPointer<vtkStructuredPointsReader> reader = vtkSmartPointer<vtkStructuredPointsReader>::New();
        reader->SetFileName(inputFilename.c_str());
        reader->Update();
        // vtkStructuredPoints is a subclass of vtkImageData, so a
        // shallow copy is enough.
        imageData->ShallowCopy(reader->GetOutput());
    }
    else
    {
        printf("Error: unsupported file extension (expected .vti or .vtk)\n");
        return;
    }

    // conversion to a Blitz++ array for easy manipulation
    int *dims = imageData->GetDimensions();
    image.resize(dims);
    int x, y, z;

    #pragma omp parallel for default(shared) private(x, y, z)
    for (z = 0; z < dims[2]; z++)
    {
        for (y = 0; y < dims[1]; y++)
        {
            for (x = 0; x < dims[0]; x++)
            {
                double *pixel = static_cast<double *>(imageData->GetScalarPointer(x, y, z));
                image(x, y, z) = pixel[0];
            }
        }
    }

    printf(" OK\n");
    printf("Dims: x: %d y: %d z: %d\n", dims[0], dims[1], dims[2]);
    printf("Number of points: %ld\n", static_cast<long>(imageData->GetNumberOfPoints()));
}

/**
 * @brief Read a binary VTK image and binarize it.
 *
 * See vtkio.h for the contract. Voxel values different from 0 are
 * treated as solid, then mapped according to porevalue.
 */
void readvtkfile_uint8(std::string &inputFilename, int & porevalue, blitz::Array<unsigned char, 3> &image)
{
    printf("Reading image file %s ...", inputFilename.c_str());

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();

    // read the file
    if (inputFilename.size() >= 3 && inputFilename.substr(inputFilename.size() - 3) == "vti")
    {
        vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
        reader->SetFileName(inputFilename.c_str());
        reader->Update();
        imageData->ShallowCopy(reader->GetOutput());
    }
    else if (inputFilename.size() >= 3 && inputFilename.substr(inputFilename.size() - 3) == "vtk")
    {
        vtkSmartPointer<vtkStructuredPointsReader> reader = vtkSmartPointer<vtkStructuredPointsReader>::New();
        reader->SetFileName(inputFilename.c_str());
        reader->Update();
        // vtkStructuredPoints is a subclass of vtkImageData, so a
        // shallow copy is enough.
        imageData->ShallowCopy(reader->GetOutput());
    }
    else
    {
        printf("Error: unsupported file extension (expected .vti or .vtk)\n");
        return;
    }

    // conversion to a Blitz++ array and binarization
    // (pixels with value > 0 are set to 1)
    int *dims = imageData->GetDimensions();
    image.resize(dims);
    uint8_t *ptr = static_cast<uint8_t *>(imageData->GetScalarPointer());
    int x, y, z;

    #pragma omp parallel for default(shared) private(x, y, z)
    for (z = 0; z < dims[2]; z++)
    {
        for (y = 0; y < dims[1]; y++)
        {
            for (x = 0; x < dims[0]; x++)
            {
                uint8_t raw = ptr[dims[0] * dims[1] * z + dims[0] * y + x];
                // Binarization: a voxel equal to porevalue is fluid.
                // The result is always stored as 0 = fluid, 1 = solid.
                image(x, y, z) = (raw == static_cast<uint8_t>(porevalue)) ? 0 : 1;
            }
        }
    }

    printf(" OK\n");
    printf("Dims: x: %d y: %d z: %d\n", dims[0], dims[1], dims[2]);
    printf("Number of points: %ld\n", static_cast<long>(imageData->GetNumberOfPoints()));
}

/* ------------------------------------------------------------------------ */
/* Writers                                                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief Write a 3D scalar array to a .vti file.
 *
 * See vtkio.h for the contract.
 */
void writeScalarImage(std::string & filename, blitz::Array<double, 3> & data)
{
    printf("Writing %s ...", filename.c_str());

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(data.shape()(0), data.shape()(1), data.shape()(2));

#if VTK_MAJOR_VERSION <= 5
    imageData->SetNumberOfScalarComponents(1);
    imageData->SetScalarTypeToDouble();
#else
    imageData->AllocateScalars(VTK_DOUBLE, 1);
#endif
    int *dims = imageData->GetDimensions();

    for (int z = 0; z < dims[2]; z++)
    {
        for (int y = 0; y < dims[1]; y++)
        {
            for (int x = 0; x < dims[0]; x++)
            {
                double *pixel = static_cast<double *>(imageData->GetScalarPointer(x, y, z));
                pixel[0] = data(x, y, z);
            }
        }
    }

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(filename.c_str());

#if VTK_MAJOR_VERSION <= 5
    writer->SetInputConnection(imageData->GetProducerPort());
#else
    writer->SetInputData(imageData);
#endif
    writer->Write();

    printf(" OK\n");
}

/**
 * @brief Write three 3D scalar arrays as a vector image to a .vti file.
 *
 * See vtkio.h for the contract.
 */
void writeVectorImage(std::string & filename, blitz::Array<double, 3> & u, blitz::Array<double, 3> & v, blitz::Array<double, 3> & w)
{
    printf("Writing %s ...", filename.c_str());

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(u.shape()(0), u.shape()(1), u.shape()(2));

#if VTK_MAJOR_VERSION <= 5
    imageData->SetNumberOfScalarComponents(3);
    imageData->SetScalarTypeToDouble();
#else
    imageData->AllocateScalars(VTK_DOUBLE, 3);
#endif
    int *dims = imageData->GetDimensions();

    for (int z = 0; z < dims[2]; z++)
    {
        for (int y = 0; y < dims[1]; y++)
        {
            for (int x = 0; x < dims[0]; x++)
            {
                double *pixel = static_cast<double *>(imageData->GetScalarPointer(x, y, z));
                pixel[0] = u(x, y, z);
                pixel[1] = v(x, y, z);
                pixel[2] = w(x, y, z);
            }
        }
    }

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(filename.c_str());

#if VTK_MAJOR_VERSION <= 5
    writer->SetInputConnection(imageData->GetProducerPort());
#else
    writer->SetInputData(imageData);
#endif
    writer->Write();

    printf(" OK\n");
}

/**
 * @brief Write a binary 3D image (0/1) to a .vti file as 8-bit data.
 *
 * See vtkio.h for the contract. The scalar type is VTK_UNSIGNED_CHAR so
 * that readvtkfile_uint8() can read the buffer back as uint8_t values.
 */
void writeBinaryImage(std::string & filename, blitz::Array<unsigned char, 3> & data)
{
    printf("Writing %s ...", filename.c_str());

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(data.shape()(0), data.shape()(1), data.shape()(2));

#if VTK_MAJOR_VERSION <= 5
    imageData->SetNumberOfScalarComponents(1);
    imageData->SetScalarTypeToUnsignedChar();
#else
    imageData->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
#endif
    int *dims = imageData->GetDimensions();

    unsigned char *ptr = static_cast<unsigned char *>(imageData->GetScalarPointer());
    for (int z = 0; z < dims[2]; z++)
    {
        for (int y = 0; y < dims[1]; y++)
        {
            for (int x = 0; x < dims[0]; x++)
            {
                ptr[dims[0] * dims[1] * z + dims[0] * y + x] = data(x, y, z);
            }
        }
    }

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(filename.c_str());

#if VTK_MAJOR_VERSION <= 5
    writer->SetInputConnection(imageData->GetProducerPort());
#else
    writer->SetInputData(imageData);
#endif
    writer->Write();

    printf(" OK\n");
}
