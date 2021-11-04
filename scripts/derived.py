#!/usr/bin/env python

# This script can be used to derive additional scalars for the Sciviscontest 2021 dataset.
# It also saves each timestep in multiple levels of detail.
# The .nc files of the dataset are expected with unchanged names inside a 'mantle' directory.
# The output will be written to a 'preprocessed' directory.

import math
import vtk

def mydot(a,b):
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def main():
    maxlevel = 3
    maxtime = 251

    for timestep in range(1, maxtime):
        reader = vtk.vtkNetCDFCFReader()
        reader.SetFileName("mantle/spherical{:03d}.nc".format(timestep))
        reader.SphericalCoordinatesOn()
        reader.Update()

        dataset = reader.GetOutput()

        radialVelocity = vtk.vtkFloatArray()
        radialVelocity.SetName("radialVelocity")
        radialVelocity.SetNumberOfComponents(1)
        radialVelocity.SetNumberOfTuples(dataset.GetNumberOfCells())

        altitude = vtk.vtkFloatArray()
        altitude.SetName("altitude")
        altitude.SetNumberOfComponents(1)
        altitude.SetNumberOfTuples(dataset.GetNumberOfCells())

        cart_X = vtk.vtkFloatArray()
        cart_X.SetName("X")
        cart_X.SetNumberOfComponents(1)
        cart_X.SetNumberOfTuples(dataset.GetNumberOfCells())

        cart_Y = vtk.vtkFloatArray()
        cart_Y.SetName("Y")
        cart_Y.SetNumberOfComponents(1)
        cart_Y.SetNumberOfTuples(dataset.GetNumberOfCells())

        cart_Z = vtk.vtkFloatArray()
        cart_Z.SetName("Z")
        cart_Z.SetNumberOfComponents(1)
        cart_Z.SetNumberOfTuples(dataset.GetNumberOfCells())

        pos = [0,0,0]
        cellVelocity = [0,0,0]

        arrayVelX = dataset.GetCellData().GetArray("vx")
        arrayVelY = dataset.GetCellData().GetArray("vy")
        arrayVelZ = dataset.GetCellData().GetArray("vz")

        for i in range(0, dataset.GetNumberOfCells()):
            cellPoints = dataset.GetCell(i).GetPoints()
            
            pos[0] = (cellPoints.GetPoint(0)[0]+cellPoints.GetPoint(1)[0]+cellPoints.GetPoint(2)[0]+cellPoints.GetPoint(3)[0]+cellPoints.GetPoint(4)[0]+cellPoints.GetPoint(5)[0]+cellPoints.GetPoint(6)[0]+cellPoints.GetPoint(7)[0] ) / 8
            pos[1] = (cellPoints.GetPoint(0)[1]+cellPoints.GetPoint(1)[1]+cellPoints.GetPoint(2)[1]+cellPoints.GetPoint(3)[1]+cellPoints.GetPoint(4)[1]+cellPoints.GetPoint(5)[1]+cellPoints.GetPoint(6)[1]+cellPoints.GetPoint(7)[1] ) / 8
            pos[2] = (cellPoints.GetPoint(0)[2]+cellPoints.GetPoint(1)[2]+cellPoints.GetPoint(2)[2]+cellPoints.GetPoint(3)[2]+cellPoints.GetPoint(4)[2]+cellPoints.GetPoint(5)[2]+cellPoints.GetPoint(6)[2]+cellPoints.GetPoint(7)[2] ) / 8

            cart_X.InsertTuple1(i, pos[0])
            cart_Y.InsertTuple1(i, pos[1])
            cart_Z.InsertTuple1(i, pos[2])

            alt = math.sqrt(mydot(pos,pos))
            altitude.InsertTuple1(i, alt)

            cellVelocity[0] = arrayVelX.GetTuple1(i) * 1e9
            cellVelocity[1] = arrayVelY.GetTuple1(i) * 1e9
            cellVelocity[2] = arrayVelZ.GetTuple1(i) * 1e9

            radialVec = (mydot(cellVelocity,pos) / alt)
            radialVelocity.InsertTuple1(i,radialVec)

            if i % 1000000 == 0 :
                print(f"{i / dataset.GetNumberOfCells()}%")

        dataset.GetCellData().AddArray(radialVelocity)
        dataset.GetCellData().AddArray(altitude)
        dataset.GetCellData().AddArray(cart_X)
        dataset.GetCellData().AddArray(cart_Y)
        dataset.GetCellData().AddArray(cart_Z)
        print(f"Writing timestep: {timestep} ")

        passArrays = vtk.vtkPassArrays()
        passArrays.SetInputData(dataset)
        passArrays.AddCellDataArray("vx")
        passArrays.AddCellDataArray("vy")
        passArrays.AddCellDataArray("vz")
        passArrays.RemoveArraysOn()
        passArrays.Update()

        for l in range(0, maxlevel):
            roi = vtk.vtkExtractGrid()
            roi.SetInputData(passArrays.GetOutput())
            roi.SetSampleRate(int(math.pow(2,l)), int(math.pow(2,l)),int(math.pow(2,l)))
            roi.IncludeBoundaryOn()
            roi.Update()

            roi.GetOutput().GetPoints().SetDataTypeToFloat()
            writer = vtk.vtkDataSetWriter()
            writer.SetFileName("preprocessed/grid_l{:01d}_t{:03d}.vtk".format(maxlevel-l-1,timestep))
            writer.SetFileTypeToBinary()
            writer.SetInputData(roi.GetOutput())
            writer.Update()


        print(f"Finished timestep: {timestep} ")
	


if __name__ == '__main__':
    main()