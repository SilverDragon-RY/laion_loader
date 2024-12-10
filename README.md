# laion_loader
Laion series are powerful dataset. However it is :
- Extremely large
- Saves URL instead of Image/Tensors
- URLs might be invalid or replaced with different images
which might influence your training.

This tool can load (n) samples from a Laion dataset, verifying URL and image integrity by its size, and store to local storage.

# Usage
```
loader = LaionBuilder(token="your_hf_token_with_access", 
                      dataset_name="any laion dataset(default:laion2B)", 
                      tgt_url="heading for URL col in dataset(default:URL)", 
                      tgt_txt="heading for text col in dataset(default:TEXT),
                      shuffle=True/False(default:True, whether to pick random n samples or first n samples
                      )           
loader.load(num_data=(how many samples you want),
            save_path="(if not provided, the name of dataset will be used)", 
            num_workers=(how many threads to use. default:16), 
            timeout=(how long to wait for a sample before calling it invalid. default:10s)
            )
```
# Format
```
- save_path
  -- random_number_1
      ---image.png
      ---text.txt
  -- random_number_2
      ---image.png
      ---text.txt
...
```
  
