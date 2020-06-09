import Minesweeper as MB

class Agent:

    def __init__(self, difficulty, controller):
        self.game = MB.Minesweeper(difficulty, (0,0,))
        self.board = game.get_board()

        self.controller = controller

        self.known_opens=[]
        self.known_mines=[]
        self.boundary=[]
        self.init_knowledge()

        #self.boundary_blocks = []

    def init_boundary(self):
        for pos in range(self.height*self.width):
            row = pos // self.width
            col = pos % self.width
            if self.board[row][col] == MB.HIDDEN:
                if self.is_boundary(row,col):
                    self.boundary.append((row,col,))
    def update_boundary(self):
        for grid in self.boundary:
            if self.board[grid[0]][grid[1]] != MB.HIDDEN:
                self.boundary.remove(grid)

        self.init_boundary()

    def is_boundary(self,row,col):
        for nrow, ncol in self.game.neighbors(row, col):
            if row != nrow and col != nrol \
                and self.board[nrow][ncol] != MB.HIDDEN:
                return True
        return False
    def boundary_split(self):
        for grid in self.boundary:
            self.boundary_blocks.append( blocks(grid[0], grid[1] ) )

    def blocks(self, row, col):
        block = []
        block.append(row,col,)
        self.boundary.remove( (row,col,) )
        for nrow, ncol in self.game.neighbors(row, col):
            if (nrow, ncol,) in self.boundary:
                block.extend( blocks(nrow, ncol) )
        return block
    def surr_flags(self,row,col):
        flags = 0
        for nrow, ncol in self.game.neighbors(row,col):
            if self.board[nrow][ncol] == MB.FLAGGED:
                flags += 1
        return hiddens
    def surr_mines(self,row,col):
        mines = 0
        for nrow,ncol in self.game.neighbors(row,col):
            if (nrow,ncol,) in self.known_mines:
                mines += 1
        return mines
    def surr_hiddens(self,row,col):
        hiddens = 0
        for nrow, ncol in self.game.neighbors(row,col):
            if self.board[nrow][ncol] == MB.HIDDEN:
                hiddens += 1
        return hiddens
    def naive_flag(self):
        for grid in self.boundary:
            for nrow,ncol in self.game.neighbors(grid[0],grid[1]):
                if self.board[nrow][ncol] != MB.HIDDEN \
                    and self.board[nrow][ncol] != MB.FLAGGED:
                    if self.board[nrow][ncol] == self.surr_hiddens(nrow,ncol):
                        for nnrow,nncol in self.game.neighbors( nrow, ncol ):
                            if self.board[nnrow][nncol] == MB.HIDDEN \
                                    and (nnrow,nncol,) not in self.known_mines:
                                self.known_mines.append( (nnrow,nncol,))

    def naive_open(self):
        for grid in self.boundary:
            for nrow, ncol in self.game.neighbors(grid[0], grid[1]):
                if self.board[nrow][ncol] != MB.HIDDEN \
                        and self.board[nrow][ncol] != MB.FLAGGED:
                    if self.board[nrow][ncol] == self.surr_flags(nrow, ncol):
                        for nnrow, nncol in self.game.neighbors(nrow, ncol):
                            if self.board[nnrow][nncol] == MB.HIDDEN \
                                    and (nnrow, nncol,) not in self.known_mines:
                                self.known_opens.append((nnrow, nncol,))
    def open_known(self):
        success = False
        for grid in self.known_opens:
            for nrow,ncol in self.game.neighbors(grid[0],grid[1]):
                if self.board[nrow][ncol] == MB.OPEN:
                    num_mines = self.board[nrow][ncol]
                    mines = self.surr_mines(nrow,ncol)
                    hiddens = self.surr_hiddens(nrow,col)

                    if num_mines == mines and hiddens > mines:
                        success = True

                        if hiddens - mines > mines:
                            for nnrow,nncol in self.game.neighbors(nrow,ncol):
                                if (nnrow,nncol,) in self.known_mines:
                                    self.controller.on_flag(nnrow,nncol)
                            self.controller.on_chord(nrow,ncol)
                            continue

                        for nnrow,nncol in self.game.neighbors(nrow,ncol):
                            if (nnrow,nncol,) in self.known_opens:
                                self.controller.on_open(nnrow,nncol)

        return success


    def solve(self):
        while True:
            if(self.game.finished()):
                break
            self.naive_flag()
            self.naive_open()
            self.open_known()
            self.update_boundary()


