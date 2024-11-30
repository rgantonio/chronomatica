import random

# Player list according to level
level_1_2 = [
    'Afsaneh',
    'Max (Italian)',
    'Joran',
    'Hoang',
    'Appo',
    'Steven (Belgian)',
]

level_3 = [
    'Julie',
    'Yiying',
    'Alex',
    'Joshua',
    'Chao',
    'Justin',
]

level_4 = [
    'Steven (Asian)',
    'Julien',
    'Anmol',
    'Venu',
    'Yong',
    'Stanley',
    'Hery',
    'Ravish',
    'Vivek',
    'Yam',
    'Hannah',
    'Lena',
]

level_5 = [
    'Mali',
    'Max (Asian)',
    'Martin',
    'Vageesh',
    'Jeremy',
    'Rofiq',
]

# Fix the random seed for reproducibility
random.seed(2024)

# Function to create randomized groups
def create_groups():
    random.shuffle(level_1_2)
    random.shuffle(level_3)
    random.shuffle(level_4)
    random.shuffle(level_5)

    groups = []
    for i in range(6):
        group = [
            level_1_2[i % len(level_1_2)],  # 1 from level_1_2
            level_3[i % len(level_3)],      # 1 from level_3
            level_4[i * 2 % len(level_4)],  # 2 from level_4
            level_4[(i * 2 + 1) % len(level_4)],
            level_5[i % len(level_5)],      # 1 from level_5
        ]
        groups.append(group)
    
    return groups

# Create and display groups
groups = create_groups()

for idx, group in enumerate(groups, start=1):
    print(f"Group {idx}: {', '.join(group)}")
