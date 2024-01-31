```

                if neighboring_cell in self.painting:
                    # append the color to the list
                    neighboring_cells_colors.append(self.board[neighboring_cell]) # board contains the colors of the cells in the board
            # 3. update the pairings dictionary
            for color in neighboring_cells_colors:
```
