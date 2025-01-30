import stow
from tqdm import tqdm
import pickle

dataset_path = stow.join('Datasets', 'IAM_Words')
words_txt_path = stow.join(dataset_path, "words.txt")

dataset, vocab, max_len = [], set(), 0

# Read and parse the words.txt file
with open(words_txt_path, "r") as file:
    words = file.readlines()

for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[1] == "err":
        continue  # Skip erroneous entries

    folder1 = line_split[0][:3]
    folder2 = line_split[0][:8]
    file_name = line_split[0] + ".png"
    label = line_split[-1].strip()

    rel_path = stow.join(dataset_path, "words", folder1, folder2, file_name)
    if not stow.exists(rel_path):
        continue  # Skip if the image file does not exist

    dataset.append([rel_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))



# Save dataset, vocab, and max_len for later use
with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump((dataset, vocab, max_len), f)

print("Preprocessing complete. Data saved.")

print(f"Dataset size: {len(dataset)}")
print(f"Unique characters in dataset: {len(vocab)}")
print(f"Max label length: {max_len}")
