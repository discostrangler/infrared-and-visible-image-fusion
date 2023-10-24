n = int(input())
powers = list(map(int, input().split()))

team1 = 0
team2 = 0
while len(powers)>0 :
    if powers[0]>powers[len(powers)-1] :
        while powers[0]>0:
            team1 += powers[0]
            powers.pop(0)
print(team1)
            
    
        
    
