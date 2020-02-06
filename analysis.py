import seaborn as sns
from matplotlib import pyplot as plt


file_path = 'train.log'

top_generation = []
top_psnr = []
top_flag = False


with open(file_path, 'r') as f:
    for line in f.readlines():
        if top_flag:
            if line.startswith('['):
                top_psnr[-1].append(float(line.split()[0][1:-1]))
            else:
                top_flag = False

        if 'Top of generation' in line:
            top_generation.append(int(line.split()[-1]))
            top_psnr.append([])
            top_flag = True


sns.boxplot(data=top_psnr, color='cyan', linewidth=0.5)
plt.xticks([], [])
plt.xlabel('Generation')
plt.ylabel('PSNR')
plt.title('Top distribution per generation')
plt.savefig(f'top.png')
