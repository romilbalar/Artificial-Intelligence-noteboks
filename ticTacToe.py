from random import randint
class Game:
	def __init__(self):
		self.board =[[None for i in range(3)]for i in range(3)]
		self.player_one, self.player_two = 5,5
		self.count = 0
		self.pawn = {"X": "Player", "O": "Computer"}
		self.winner = False

	def printBoard(self):
		print()
		for i in range(3):
			print(self.board[i])

	def checkFilled(self,row,col):
		return self.board[row][col]

	def computer(self):
		row = randint(0,2)
		col = randint(0,2)
		if self.board[row][col] != None:
			self.computer()
		else:
			self.board[row][col] = "0"
			if self.check(row,col):
				print("Computer Wins")

	def player(self):
		row = int(input("Row : "))
		col = int(input("Col : "))
		if not self.checkFilled(row,col):
			self.board[row][col] = "X"
			if self.check(row,col):
				print("Player Wins")
		else:
			print("Position is not empty... Try again")
			self.player()

	def place(self):
		if self.count < 9:
			if self.count % 2 == 0:
				self.player()
			else:
				self.computer()

			self.count += 1
			self.printBoard()


	def check(self,row,col):
		if self.board[row][0] == self.board[row][1] ==  self.board[row][2]:
			self.winner = True
			return True
		elif self.board[0][col] == self.board[1][col] ==  self.board[2][col]:
			self.winner = True
			return True
		elif row == col and self.board[0][0] == self.board[1][1] ==  self.board[2][2]:
			self.winner = True
			return True
		elif row + col == 2 and self.board[2][0] == self.board[1][1] ==  self.board[0][2]:
			self.winner = True
			return True


	def play(self):
		print("Player : X")
		print("Computer : O")
		for i in range(9):
			if self.count<9 and not self.winner:
				self.place()
			else:
				break
obj = Game()
obj.play()
