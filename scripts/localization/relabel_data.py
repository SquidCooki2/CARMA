import os

def relabel_to_single_class(base_path):
    # The folders to process
    splits = ['train', 'valid', 'test']
    
    count = 0
    for split in splits:
        labels_dir = os.path.join(base_path, split, 'labels')
        if not os.path.exists(labels_dir):
            print(f"Directory not found: {labels_dir}")
            continue
            
        print(f"Processing {split} labels...")
        
        for filename in os.listdir(labels_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(labels_dir, filename)
                
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                new_lines = []
                for line in lines:
                    parts = line.split()
                    if len(parts) > 0:
                        # Force the first part (class ID) to '0'
                        parts[0] = '0'
                        new_lines.append(" ".join(parts) + "\n")
                
                with open(file_path, 'w') as f:
                    f.writelines(new_lines)
                
                count += 1
    
    print(f"Finished. Relabeled {count} files to class 0.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    egohands_path = os.path.join(script_dir, "../../data/egohands")
    relabel_to_single_class(egohands_path)
