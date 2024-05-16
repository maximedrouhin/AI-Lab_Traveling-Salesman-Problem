# %% [markdown]
# # Setup file

# %% [markdown]
# This files lets you download and extract the data from Kaggle for this project.

# %% [markdown]
# ## Prerequisites

# %% [markdown]
# - Ensure you have a Kaggle account:
#   https://www.kaggle.com/account/login?phase=startRegisterTab
# - Download your Kaggle API credentials and place them in the right directory:
#   https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials
# - Accept the competition rules to be able to download the data:
#   https://www.kaggle.com/competitions/tsp-cv/rules

# %% [markdown]
# ## Setup code

# %%
%pip install kaggle

# %%
!kaggle competitions download -c tsp-cv

# %%
# unzip tsp-cv.zip using python
import zipfile

with zipfile.ZipFile('tsp-cv.zip', 'r') as zip_ref:
    zip_ref.extractall('tsp-cv')

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create a figure with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

fig.set_facecolor('black')

# Loop through the image files and display them in the subplots
for i in range(6):
    # Calculate the row and column index
    row = i // 3
    col = i % 3
    
    # Load and display the image
    img = mpimg.imread(f'tsp-cv/{i}.jpg')
    axs[row, col].imshow(img)
    axs[row, col].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create a figure with 3 rows and 3 columns
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

fig.set_facecolor('black')

# Loop through the image files and display them in the subplots
for i in range(3):
    # Calculate the row and column index

    # Load and display the image
    img = mpimg.imread(f'tsp-cv/{i}.jpg')

    for col in range(3):
        axs[i, col].imshow(img[:,:,col].repeat(3).reshape(img.shape))
        axs[i, col].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()


