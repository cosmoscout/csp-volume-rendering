#!/usr/bin/env python

# This script can be used to generate a csv file for the parallel coordinates plot from a NetCDF file.
# The file to be converted has to be given as the first command line argument.
# The resulting csv file will be saved next to the original file.

import math
import sys
import os
import vtk

def main(path):
    reader = vtk.vtkNetCDFCFReader()
    reader.SetFileName(path)
    reader.SphericalCoordinatesOff()
    reader.Update()
    
    resample = vtk.vtkResampleToImage()
    resample.SetInputConnection(reader.GetOutputPort())
    resample.SetSamplingDimensions(40, 40, 40)
    resample.SetUseInputBounds(1)
    resample.Update()
    
    pointToCell = vtk.vtkPointDataToCellData()
    pointToCell.SetInputData(resample.GetOutput())
    pointToCell.Update()
    
    passArrays = vtk.vtkPassArrays()
    passArrays.SetInputData(pointToCell.GetOutput())
    passArrays.AddCellDataArray("vtkGhostType")
    passArrays.AddCellDataArray("vtkValidPointMask")
    passArrays.RemoveArraysOn()
    passArrays.Update()
    
    objToTable = vtk.vtkDataObjectToTable()
    objToTable.SetInputData(passArrays.GetOutput())
    objToTable.SetFieldType(2)
    objToTable.Update()
    
    csvwriter = vtk.vtkDelimitedTextWriter()
    csvwriter.SetInputData(objToTable.GetOutput())
    split = os.path.splitext(path)
    csvwriter.SetFileName(split[0] + ".csv")
    csvwriter.Update()


if __name__ == '__main__':
    main(sys.argv[1])