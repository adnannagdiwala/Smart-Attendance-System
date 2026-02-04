import os

dataset_path = "dataset"

print("Fixing file extensions...\n")

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    if os.path.isdir(person_path):
        print(f"Processing {person}...")
        
        for file in os.listdir(person_path):
            file_path = os.path.join(person_path, file)
            
            # Get extension
            name, ext = os.path.splitext(file)
            ext_lower = ext.lower()
            
            # If it's an image but not .jpg
            if ext_lower in ['.jpeg', '.png', '.heic', '.webp'] and ext != '.jpg':
                new_file_path = os.path.join(person_path, name + '.jpg')
                
                # If it's PNG or HEIC, convert to JPG
                if ext_lower in ['.png', '.heic', '.webp']:
                    try:
                        from PIL import Image
                        img = Image.open(file_path)
                        # Convert to RGB if needed
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(new_file_path, 'JPEG')
                        os.remove(file_path)
                        print(f"   ✓ Converted {file} to .jpg")
                    except Exception as e:
                        print(f"   ✗ Error converting {file}: {e}")
                
                # If it's JPEG, just rename
                elif ext_lower == '.jpeg':
                    os.rename(file_path, new_file_path)
                    print(f"   ✓ Renamed {file} to {name}.jpg")

print("\n✓ Done!")