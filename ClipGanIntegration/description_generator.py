import random

# Lists of potential details
male_first_names = [
    "John", "Michael", "James", "David", "Robert",
    "William", "Joseph", "Charles", "Thomas", "Christopher",
    "Daniel", "Matthew", "Anthony", "Mark", "Donald",
    "Steven", "Paul", "Andrew", "Joshua", "Kenneth",
    "Richard", "Kevin", "Brian", "Edward", "George",
    "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob",
    "Gary", "Nicholas", "Eric", "Jonathan", "Stephen",
    "Larry", "Scott", "Frank", "Justin", "Brandon",
    "Raymond", "Gregory", "Benjamin", "Samuel", "Patrick",
    "Alexander", "Jack", "Dennis", "Jerry", "Tyler",
    "Aaron", "Henry", "Douglas", "Peter", "Adam",
    "Nathan", "Zachary", "Kyle", "Walter", "Harold",
    "Carl", "Arthur", "Dylan", "Bryan", "Joe",
    "Jordan", "Albert", "Gabriel", "Randy", "Louis",
    "Philip", "Harry", "Logan", "Bruce", "Russell",
    "Bobby", "Johnny", "Phillip", "Eugene", "Ralph",
    "Ronald", "Christian", "Lawrence", "Austin", "Roger",
    "Alan", "Shawn", "Jesse", "Ethan", "Wayne",
    "Vincent", "Martin", "Roy", "Billy", "Willie",
    "Curtis", "Sean", "Jeremy", "Larry", "Jose",
    "Terry", "Eddie", "Ricky"
]

female_first_names = [
    "Mary", "Emily", "Sarah", "Jessica", "Jennifer",
    "Elizabeth", "Linda", "Susan", "Margaret", "Dorothy",
    "Lisa", "Nancy", "Karen", "Betty", "Helen",
    "Sandra", "Ashley", "Donna", "Kimberly", "Patricia",
    "Megan", "Laura", "Michelle", "Carol", "Amanda",
    "Melissa", "Stephanie", "Rebecca", "Sharon", "Cynthia",
    "Kathleen", "Amy", "Angela", "Deborah", "Jessica",
    "Shirley", "Catherine", "Christine", "Rachel", "Janet",
    "Maria", "Heather", "Diane", "Julie", "Kelly",
    "Denise", "Brenda", "Catherine", "Katherine", "Diana",
    "Pamela", "Theresa", "Jane", "Beverly", "Alice",
    "Jacqueline", "Cheryl", "Martha", "Rose", "Anne",
    "Jean", "Joan", "Victoria", "Ruby", "Christina",
    "Ann", "Lori", "Julia", "Olivia", "Marie",
    "Madison", "Frances", "Hannah", "Elaine", "Gloria",
    "Marilyn", "Teresa", "Sara", "Janice", "Judith",
    "Emma", "Amber", "Brittany", "Doris", "Marilyn",
    "Danielle", "Holly", "Natalie", "Charlotte", "Sophia",
    "Grace", "Evelyn", "Madeline", "Beverly"
]

last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris",
    "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright",
    "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall",
    "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
    "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes",
    "Stewart", "Morris", "Morales", "Murphy", "Cook",
    "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper",
    "Peterson", "Bailey", "Reed", "Kelly", "Howard",
    "Ramos", "Kim", "Cox", "Ward", "Richardson",
    "Watson", "Brooks", "Chavez", "Wood", "James",
    "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
    "Price", "Alvarez", "Castillo", "Sanders", "Patel",
    "Myers", "Long", "Ross", "Foster", "Jimenez",
    "Powell", "Jenkins", "Perry", "Russell", "Sullivan",
    "Bell", "Coleman", "Butler", "Henderson", "Barnes",
    "Gonzales", "Fisher", "Vasquez", "Simmons", "Romero",
    "Jordan", "Patterson", "Alexander", "Hamilton", "Graham",
    "Reynolds", "Griffin", "Wallace"
]

occupations = ["teacher", "engineer", "doctor", "artist", "lawyer"]
hometowns = ["St. Louis, Missouri", "New York City, New York", "Los Angeles, California", "Austin, Texas", "Chicago, Illinois"]
marital_statuses = ["single", "married", "divorced"]
children_options = ["no kids", "one child", "two children", "three children"]

# List of physical traits
ages_child = list(range(5, 13))
ages_teen = list(range(13, 18))
ages_young_adult = list(range(18, 30))
ages_adult = list(range(30, 50))
ages_middle_aged = list(range(50, 65))
ages_elderly = list(range(65, 90))

# Function to parse the user prompt
def parse_prompt(prompt):
    details = {}
    prompt = prompt.lower()

    # Check for female-related terms first
    if "woman" in prompt or "gal" in prompt or "girl" in prompt:
        details["gender"] = "female"
        details["age_group"] = "child" if "girl" in prompt else "adult"
    # Check for male-related terms second
    elif "man" in prompt or "guy" in prompt or "boy" in prompt:
        details["gender"] = "male"
        details["age_group"] = "child" if "boy" in prompt else "adult"
    # Handle generic term 'person'
    elif "person" in prompt:
        details["gender"] = random.choice(["male", "female"])
        details["age_group"] = random.choice(["child", "adult"])

    # Recognize specific age terms
    if "child" in prompt:
        details["age_group"] = "child"
    elif "young" in prompt or "teen" in prompt:
        details["age_group"] = "teen"
    elif "young adult" in prompt:
        details["age_group"] = "young_adult"
    elif "middle aged" in prompt:
        details["age_group"] = "middle_aged"
    elif "elderly" in prompt or "old" in prompt:
        details["age_group"] = "elderly"
    
    return details

# Generate random details based on the prompt
def generate_random_persona(prompt):
    details = parse_prompt(prompt)
    
    gender = details.get("gender", random.choice(["male", "female"]))
    age_group = details.get("age_group", random.choice(["child", "teen", "young_adult", "adult", "middle_aged", "elderly"]))
    
    if gender == "male":
        first_name = random.choice(male_first_names)
    else:
        first_name = random.choice(female_first_names)
    
    last_name = random.choice(last_names)
    
    if age_group == "child":
        age = random.choice(ages_child)
        occupation = "student"
        marital_status = "single"
        children = "no kids"
    elif age_group == "teen":
        age = random.choice(ages_teen)
        occupation = "student"
        marital_status = "single"
        children = "no kids"
    elif age_group == "young_adult":
        age = random.choice(ages_young_adult)
        occupation = random.choice(occupations)
        marital_status = random.choice(marital_statuses)
        children = random.choice(children_options)
    elif age_group == "adult":
        age = random.choice(ages_adult)
        occupation = random.choice(occupations)
        marital_status = random.choice(marital_statuses)
        children = random.choice(children_options)
    elif age_group == "middle_aged":
        age = random.choice(ages_middle_aged)
        occupation = random.choice(occupations)
        marital_status = random.choice(marital_statuses)
        children = random.choice(children_options)
    else:  # elderly
        age = random.choice(ages_elderly)
        occupation = random.choice(occupations)
        marital_status = random.choice(marital_statuses)
        children = random.choice(children_options)
    
    hometown = random.choice(hometowns)
    
    # Create the description
    description = (f"Name: {first_name} {last_name}\n"
                   f"Age: {age}\n"
                   f"Occupation: {occupation}\n"
                   f"Hometown: {hometown}\n"
                   f"Marital Status: {marital_status}\n"
                   f"Children: {children}\n"
                   f"Personality Traits: {random.choice(['kind', 'wise', 'funny', 'serious', 'thoughtful'])}, "
                   f"{random.choice(['patient', 'energetic', 'creative', 'analytical', 'resourceful'])}")
    return description

# User prompt
user_prompt = input("Enter a description for the persona: ")
# Generate and print a random persona based on the prompt
print(generate_random_persona(user_prompt))
