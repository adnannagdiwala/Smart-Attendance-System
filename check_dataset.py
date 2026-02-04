import os

dataset_path = "dataset"

print("Checking dataset folder structure...\n")
print("=" * 60)

if not os.path.exists(dataset_path):
    print("❌ Dataset folder doesn't exist!")
else:
    people_folders = os.listdir(dataset_path)
    
    if len(people_folders) == 0:
        print("❌ Dataset folder is empty!")
    else:
        for person in people_folders:
            person_path = os.path.join(dataset_path, person)
            
            if os.path.isdir(person_path):
                print(f"\n📁 Folder: {person}")
                print("-" * 60)
                
                all_files = os.listdir(person_path)
                
                if len(all_files) == 0:
                    print("   ⚠️  Folder is empty!")
                else:
                    print(f"   Total files: {len(all_files)}")
                    print("\n   Files found:")
                    
                    for i, file in enumerate(all_files[:10], 1):  # Show first 10
                        file_path = os.path.join(person_path, file)
                        if os.path.isfile(file_path):
                            # Get file extension
                            ext = os.path.splitext(file)[1].lower()
                            print(f"   {i}. {file} (Extension: {ext})")
                    
                    if len(all_files) > 10:
                        print(f"   ... and {len(all_files) - 10} more files")
                    
                    # Count by extension
                    extensions = {}
                    for file in all_files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext:
                            extensions[ext] = extensions.get(ext, 0) + 1
                    
                    print(f"\n   Extension breakdown:")
                    for ext, count in extensions.items():
                        print(f"   {ext}: {count} files")

print("\n" + "=" * 60)