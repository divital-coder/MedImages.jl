
module Load_and_save
using Dictionaries, Dates
using Accessors, UUIDs, ITKIOWrapper
using ..MedImage_data_struct
using ..MedImage_data_struct:MedImage
using ..Brute_force_orientation
using ..Utils
export load_image
export update_voxel_and_spatial_data



"""
helper function for nifti #1
Determines the study type for an image based on its metadata using SimpleITK
"""
#function infer_modality(image)
#    sitk = pyimport("SimpleITK")
#    metadata = Dict()
#    for key in image.GetMetaDataKeys()
#        metadata[key] = image.GetMetaData(key)
#    end
#
#    # Check for CT-specific metadata
#    if "modality" in keys(metadata) && metadata["modality"] == "CT"
#        return "CT"
#    end
#
#    spacing = image.GetSpacing()
#    size = image.GetSize()
#    pixel_type = image.GetPixelIDTypeAsString()
#
#    # Relaxed spacing condition for CT
#    spacing_condition_ct = all(s -> s <= 1.0, spacing)
#    pixel_type_condition_ct = pixel_type == "16-bit signed integer"
#
#    if spacing_condition_ct && pixel_type_condition_ct
#        # Additional check: sample some voxels to see if they're in the typical CT range
#        stats = sitk.StatisticsImageFilter()
#        stats.Execute(image)
#        mean_value = stats.GetMean()
#
#        if -1000 <= mean_value <= 3000  # Typical range for CT in Hounsfield units
#            return MedImage_data_struct.CT_type
#        end
#    end
#
#    # PET condition (unchanged)
#    spacing_condition_pet = all(s -> s > 1.0, spacing)
#    pixel_type_condition_pet = occursin("float", lowercase(pixel_type))
#
#    if spacing_condition_pet && pixel_type_condition_pet
#        return MedImage_data_struct.PET_type
#    end
#
#    return MedImage_data_struct.CT_type
#end
function infer_modality(modality::String)
  return uppercase(modality) == "CT" ?  MedImage_data_struct.CT_type : MedImage_data_struct.PET_type 
end


function load_images(path::String, modality::String)::Array{MedImage}
     #defaulting to LPS orientation for all image
    spatial_meta = ITKIOWrapper.loadSpatialMetaData(path)
    origin = spatial_meta.origin
    spacing = spatial_meta.spacing 
    direction = spatial_meta.direction
    size = spatial_meta.size

    voxel_arr = ITKIOWrapper.loadVoxelData(path,spatial_meta)
    voxel_arr = voxel_arr.dat
    voxel_arr = permutedims(voxel_arr, (1, 2, 3))
    spatial_metadata_keys = ["origin", "spacing", "direction"]
    spatial_metadata_values = [origin, spacing, direction]
    spatial_metadata = Dictionaries.Dictionary(spatial_metadata_keys, spatial_metadata_values)
    legacy_file_name_field = string(split(path, "/")[length(split(path, "/"))])

    return [MedImage(voxel_data=voxel_arr, origin=origin, spacing=spacing, direction=direction, patient_id="test_id", image_type=infer_modality(modality), image_subtype=MedImage_data_struct.CT_subtype, legacy_file_name=legacy_file_name_field)]


end



function create_nii_from_medimage(med_image::MedImage, file_path::String)
  # Convert voxel_data to a numpy array (Assuming voxel_data is stored in Julia array format)
  voxel_data_np = med_image.voxel_data
  voxel_data_np = permutedims(voxel_data_np, (3, 2, 1))
   #Create a SimpleITK image from numpy array
  sitk = pyimport("SimpleITK")
  image_sitk = sitk.GetImageFromArray(voxel_data_np)

  # Set spatial metadata
  image_sitk.SetOrigin(med_image.origin)
  image_sitk.SetSpacing(med_image.spacing)
  image_sitk.SetDirection(med_image.direction)

  # Save the image as .nii.gz
  sitk.WriteImage(image_sitk, file_path * ".nii.gz")
end



function update_voxel_data(old_image, new_voxel_data::AbstractArray)

  return MedImage(
    new_voxel_data,
    old_image.origin,
    old_image.spacing,
    old_image.direction,
    instances(Image_type)[Int(old_image.image_type)+1],#  Int(image.image_type),
    instances(Image_subtype)[Int(old_image.image_subtype)+1],#  Int(image.image_subtype),
    # old_image.voxel_datatype,
    old_image.date_of_saving,
    old_image.acquistion_time,
    old_image.patient_id,
    instances(current_device_enum)[Int(old_image.current_device)+1], #Int(image.current_device),
    old_image.study_uid,
    old_image.patient_uid,
    old_image.series_uid,
    old_image.study_description,
    old_image.legacy_file_name,
    old_image.display_data,
    old_image.clinical_data,
    old_image.is_contrast_administered,
    old_image.metadata)

end

function update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray
  , new_origin, new_spacing, new_direction)

  return MedImage(
    new_voxel_data,
    new_origin,
    new_spacing,
    new_direction,
    # old_image.spatial_metadata,
    instances(Image_type)[Int(old_image.image_type)+1],#  Int(image.image_type),
    instances(Image_subtype)[Int(old_image.image_subtype)+1],#  Int(image.image_subtype),
    # old_image.voxel_datatype,
    old_image.date_of_saving,
    old_image.acquistion_time,
    old_image.patient_id,
    instances(current_device_enum)[Int(old_image.current_device)+1], #Int(image.current_device),
    old_image.study_uid,
    old_image.patient_uid,
    old_image.series_uid,
    old_image.study_description,
    old_image.legacy_file_name,
    old_image.display_data,
    old_image.clinical_data,
    old_image.is_contrast_administered,
    old_image.metadata)

end

# function update_voxel_and_spatial_data(old_image, new_voxel_data::AbstractArray, new_origin, new_spacing, new_direction)

#   res = @set old_image.voxel_data = new_voxel_data
#   res = @set res.origin = Utils.ensure_tuple(new_origin)
#   res = @set res.spacing = Utils.ensure_tuple(new_spacing)
#   res = @set res.direction = Utils.ensure_tuple(new_direction)
#   # voxel_data=new_voxel_data
#   # origin=new_origin
#   # spacing=new_spacing
#   # direction=new_direction

#   # return @pack! old_image = voxel_data, origin, spacing, direction
#   return res
# end

"""
load image from path
"""
function load_image(path, modality)
  # test_image_equality(p,p)

  medimage_instance_array = load_images(path, modality)
  medimage_instance = medimage_instance_array[1]
  return medimage_instance
end#load_image

end




