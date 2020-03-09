def find_index_0(num):
	if len(str(num))==8:
		return 0
	for i in range(9):
		x = num % 10
		num = int(num/10)
		if x==0:
			break
	return 8-i
	
def swap(strr, a, b):
	t = list(strr)
	temp = t[a]
	t[a] = t[b]
	t[b] = temp
	
	return ''.join(t)

def make_move(num):
	ind = find_index_0(num)
	operate = str(num)
	moved_list = []
	if len(operate)==8:
		operate = '0' + operate
	if ind==0:
		final = swap(operate, ind, 1)			
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 3)
		moved_list.append([int(final),num])	
	elif ind==1:
		final = swap(operate, ind, 0)		
		final = final[1:]
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 2)
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 4)
		moved_list.append([int(final),num])
	elif ind==2:
		final = swap(operate, ind, 1)			
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 5)
		moved_list.append([int(final),num])
	elif ind==3:
		final = swap(operate, ind, 0)			
		final = final[1:]
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 4)
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 6)
		moved_list.append([int(final),num])
	elif ind==4:
		final = swap(operate, ind, 1)			
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 3)
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 5)
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 7)
		moved_list.append([int(final),num])
	elif ind==5:
		final = swap(operate, ind, 2)			
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 4)
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 8)
		moved_list.append([int(final),num])
	elif ind==6:
		final = swap(operate, ind, 3)		
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 7)
		moved_list.append([int(final),num])
	elif ind==7:
		final = swap(operate, ind, 4)		
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 6)
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 8)
		moved_list.append([int(final),num])
	elif ind==8:
		final = swap(operate, ind, 5)			
		moved_list.append([int(final),num])
		
		final = swap(operate, ind, 7)
		moved_list.append([int(final),num])

	return moved_list


print("\nEnter a 9 digit number showing the orientation of the game (row wise)")
number = int(input())

Q = []								
Q.append([number,111111111])		

moves = 0							
nodes = 0							
last = number

while Q:
	x = Q.pop(0)					
	
	nodes+=1
	
	if x[0]==last:
		moves+=1
	
	listing = make_move(x[0])		
	
	for i in listing:				
		if i[0]==x[1]:
			listing.remove(i)
			break
	
	if listing[len(listing)-1][1]==last:			
		last = 	listing[len(listing)-1][0]	
	
	for i in listing:
		if i[0] == 123456780:						
			print("Number of moves - " + str(moves+1))
			print("Number of nodes - " + str(nodes))
			print("Solution Found!")
			exit(0)
		Q.append(i)	