import os
import shutil


def move_files_to_main_directory(main_dir):
	# Iterate through all items in the main directory
	for item in os.listdir(main_dir):
		item_path = os.path.join(main_dir, item)

		# Check if the item is a directory
		if os.path.isdir(item_path):
			# Iterate through all files in the subdirectory
			for file in os.listdir(item_path):
				file_path = os.path.join(item_path, file)

				# Check if it's a file (not a directory)
				if os.path.isfile(file_path):
					# Generate the destination path in the main directory
					destination = os.path.join(main_dir, file)

					# If a file with the same name already exists, add a suffix
					counter = 1
					while os.path.exists(destination):
						print(f"File: {destination} already exists.")
						name, ext = os.path.splitext(file)
						destination = os.path.join(main_dir, f"{name}_{counter}{ext}")
						counter += 1

					# Move the file
					shutil.move(file_path, destination)
					print(f"Moved: {file} to {destination}")

			# Remove the now empty subdirectory
			os.rmdir(item_path)
			print(f"Removed empty directory: {item_path}")


# Example usage
main_directory = "home/ha51dybi/PEBAL/cityscapes/annotation/city_gt_fine/train"
move_files_to_main_directory(main_directory)