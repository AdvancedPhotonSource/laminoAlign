import vtk
import vtk.util
import vtk.util.numpy_support


def saveAsVTK(inputArray, savePath):
    """Convert 3D NumPy array to VTK array"""
    
    vtk_data_array = vtk.util.numpy_support.numpy_to_vtk(
        inputArray.ravel(), deep=True, array_type=vtk.VTK_FLOAT)

    # Create a VTK image data object
    image_data = vtk.vtkImageData()

    # Set the dimensions with the correct order (Fortran order)
    image_data.SetDimensions(inputArray.shape[::-1])

    image_data.GetPointData().SetScalars(vtk_data_array)

    # Write to VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(savePath)
    writer.SetInputData(image_data)
    writer.Write()
