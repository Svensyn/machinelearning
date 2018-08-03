import random

#dnd d20 roller

def random_number_generator(upperlimit):
    rng = random.randint (1,upperlimit)
    return rng
def d20roll ():
    roll20 = random_number_generator(20)
    return roll20

"""if d20 == 20:
    print("A crit!")
elif d20 == 1:
    print("A failure!")
else:
    print("Good hit!")"""
#dnd d10 hit power and healing power
def d10roll ():
    roll10 = random_number_generator(10)
    return roll10

#health gained by drinking a potion

def potions ():
    potionhealing = d20roll()
    return  potionhealing
#attack functions
def accuracycheck(d20):
    hit = False
    if d20 == 20:
        print("A critical hit!")
        print("")
        hit = True
    elif d20 >= 6:
        print("A hit!")
        print("")
        hit = True
    else:
        print("A miss!")
        print("")
    return hit

def humanattack (d20):
    hit = accuracycheck(d20)
    if hit == True and d20 == 20:
        damage = d10roll() * 2
    else:
        damage = d10roll()

    return damage




def orcattack ():
    hit = accuracycheck(d20)
    if hit == True and d20 == 20:
        damage = (d10roll() + 1) * 2
    else:
        damage = d10roll() + 1

    return damage


yourpotions = 2
orcpotions = 1
orchealth = 25
yourhealth = 25

while orchealth > 0 and yourhealth > 0:
    action = input("Do you want to drink a potion (P) or attack (anything else)? ")
    if yourpotions > 0 and action == "P":

            heal = potions()
            yourhealth += heal
            print ("You gained", heal, "health!")
            print ("You now have", yourhealth, "health!")
            print ("")
            yourpotions -= 1
    elif yourpotions == 0 and action == "P":
        print("You reach for a potion, but you realize you have none left!")

    else:
        print("You attack!")
        damage = humanattack(d20roll())
        if d20 >= 6:
            orchealth -= damage
            print("You deal", damage, "to the orc!")
            print("The orc now has", orchealth, "health!")
            print("")

    if orchealth < 15 and orcpotions >  0:
        heal = potions()
        orchealth += heal
        print("The orc drinks a potion to heal!")
        orcpotions -= 1
        print("The orc gained", heal, "health!")
        print("The orc now has", orchealth, "health!")
        print("")
    else:
        d20 = d20roll()
        print("The orc attacks!")
        if d20 >= 8:
            damage = orcattack()
            yourhealth -= damage
            print("The orc does", damage, "damage to you!")
            print("You now have", yourhealth, "health!")
            print("")
        else:
            print ("The orc missed!")
            print("")
if orchealth <= 0 and yourhealth <= 0:
    print ("A tie!")
elif orchealth <= 0:
    print ("You win!")
else:
    print ("You lose!")