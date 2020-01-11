#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <algorithm>
#include <ctime>
#include <fstream>
#include <string>

using namespace std;

enum Player : int {WHITE=1, BLACK=-1, DRAW=0};

class Piece
{
	char get_piece_type(int i)
	{
		switch (i)
		{
			case 0:
				return ' ';
			case 1:
			case 8:
			case 17:
			case 24:
				return 'R';
			case 2:
			case 7:
			case 18:
			case 23:
				return 'N';
			case 3:
			case 6:
			case 19:
			case 22:
				return 'B';
			case 4:
			case 20:
				return 'Q';
			case 5:
			case 21:
				return 'K';
			default:
				return 'P';
		}
	}
public:
	char type;
	Player player;
	int id, x, y, times_moved;
	bool captured, enpassantable;

	Piece(int _id): id(_id) {
		type = get_piece_type(id);
	}
	Piece(int _id, int i, int j)
		: id(_id), x(i), y(j)
	{
		type = get_piece_type(id);
		if (id >= 17)
			player = BLACK;
		else
			player = WHITE;
		times_moved = 0;
		captured = false;
		enpassantable = false;
	}
};

struct Move
{
	int index, x_from, y_from, x_to, y_to;
	char promote_from, promote_to;
	bool become_enpassantable;
	struct Capture
	{
		int index, x, y;
	} capture;

	Move(const Piece* piece, int _x_to, int _y_to, char _promote_to,
		int _capture_index, int _capture_x, int _capture_y
	) :
		x_to(_x_to), y_to(_y_to), promote_to(_promote_to)
	{
		index = piece->id;
		x_from = piece->x;
		y_from = piece->y;

		promote_from = piece->type;

		capture.index = _capture_index;
		capture.x = _capture_x;
		capture.y = _capture_y;

		become_enpassantable = (
			piece->type == 'P' and
			abs(y_to - y_from) == 2
		);
	}
};

class Board
{
	Player to_move;
	vector<vector<Piece*>> board;
	vector<Piece*> pieces;
	Piece* empty;
	vector<Move*> history;
public:
	Board()
	{
		empty = new Piece(0);
		pieces.push_back(empty);
		board = vector<vector<Piece*>>(8, vector<Piece*>(8, empty));

		vector<vector<int>> ranks = {
			// {first piece index on the row, row number}
			{1,  0},
			{9,  1},
			{17, 7},
			{25, 6},
		};
		for (const auto& rank : ranks)
		{
			for (auto i = 0; i < 8; i++)
			{
				const auto& id = rank[0]+i;
				const auto& piece = new Piece(id, i, rank[1]);
				board[rank[1]][i] = piece;
				pieces.push_back(piece);
			}
		}

		to_move = WHITE;
	}

	~Board()
	{
		for (auto& piece : pieces) {
			delete piece;
		}
	}

	bool in_check(Player player)
	{
		switch (player)
		{
			case WHITE:
				return not reached_by(pieces[5]->x, pieces[5]->y, BLACK, true).empty();
			case BLACK:
				return not reached_by(pieces[21]->x, pieces[21]->y, WHITE, true).empty();
		}
	}

	Piece* piece_at(int x, int y)
	{
		if (x < 0 or x > 7 or y < 0 or y > 7)
			return pieces[0];

		return pieces[board[y][x]->id];
	}

	vector<Piece*> reached_by(const int x, const int y, const Player& player, const bool capture)
	{
		vector<Piece*> pieces;

		if (not capture and board[y][x]->id)
			return pieces;

		vector<vector<int>> knight_moves = {{-2, 1},{-2,-1},{-1, 2},{-1,-2},{ 1, 2},{ 1,-2},{ 2, 1},{ 2,-1}};
		vector<vector<int>> king_moves   = {{ 0, 1},{ 1, 1},{ 1, 0},{ 1,-1},{ 0,-1},{-1,-1},{-1, 0},{-1, 1}};

		for (const auto& move : knight_moves) {
			const auto& piece = piece_at(x+move[0], y+move[1]);
			if (piece->type == 'N' and piece->player == player)
				pieces.push_back(piece);
		}

		for (const auto& move : king_moves) {
			const auto& piece = piece_at(x+move[0], y+move[1]);
			if (piece->type == 'K' and piece->player == player)
				pieces.push_back(piece);
		}

		for (const auto& direction : king_moves) {
			int _x = x, _y = y;
			for(;;) {
				_x += direction[0];
				_y += direction[1];

				if (_x < 0 or _x > 7 or _y < 0 or _y > 7)
					break;

				const auto& piece = piece_at(_x, _y);

				switch (piece->type) {
					case ' ':
						continue;
					case 'Q':
						if (piece->player == player)
							pieces.push_back(piece);
						break;
					case 'B':
						if (
							piece->player == player and
							direction[0] != 0 and
							direction[1] != 0
						)
							pieces.push_back(piece);
						break;
					case 'R':
						if (
							piece->player == player and
							(
								direction[0] == 0 or
								direction[1] == 0
							)
						)
							pieces.push_back(piece);
						break;
				}
				break;
			}
		}

		if (capture) {
			for (int i=-1; i<2; i+=2) {
				Piece* piece = piece_at(x+i, y-player);
				if (
					piece->type == 'P' and
					piece->player == player
				)
					pieces.push_back(piece);
			}
		} else {
			Piece* piece = piece_at(x, y-player);
			if (
				piece->type == 'P' and
				piece->player == player
			)
				pieces.push_back(piece);
			else if (piece->type == ' ')
			{
				piece = piece_at(x, y-player*2);
				if (
					piece->type == 'P' and
					piece->player == player and
					not piece->times_moved
				)
					pieces.push_back(piece);
			}
		}

		return pieces;
	}

	Player opponent()
	{
		return (to_move == WHITE) ? BLACK : WHITE;
	}

	set<Move*> legal_moves()
	{
		set<Move*> moves;

		for (int i=0; i<8; ++i) {
			for (int j=0; j<8; ++j) {
				const auto& square = board[j][i];
				if (
					square->type != ' ' and
					square->player == to_move
				)
					continue;

				const auto& capture = (square->type != ' ');
				for (const auto& piece : reached_by(i, j, to_move, capture)) {
					int id = piece->id < 17 ? piece->id : piece->id-16;
					vector<char> promotions;
					if (
						piece->type == 'P' and (
							j == 0 or
							j == 7
						)
					) {
						promotions = vector<char>({'R', 'N', 'B', 'Q'});
					} else {
						promotions.push_back(piece->type);
					}

					for (const auto promotion : promotions) {
						Move* move = new Move(
							piece, i, j, promotion, square->id, i, j
						);
						apply_move(move);
						if (in_check(piece->player)) {
							undo_move(move);
							continue;
						}
						moves.insert(move);
						undo_move(move);
					}
				}
			}
		}

		// en passant
		for (const auto& target : pieces) {
			if (
				target->enpassantable and
				target->player == opponent()
			) {
				vector<Piece*> attackers;
				if (target->x > 0) attackers.push_back(board[target->y][target->x-1]);
				if (target->x < 7) attackers.push_back(board[target->y][target->x+1]);
				for (const auto& attacker : attackers) {
					if (
						attacker->type == 'P' and
						attacker->player == to_move
					) {
						moves.insert(new Move(attacker, target->x, target->y-to_move, 'P', target->id, target->x, target->y));
					}
				}
			}
		}

		Piece* king, *rook_k, *rook_q;
		int rank;
		switch (to_move)
		{
			case WHITE:
				king = pieces[5];
				rook_k = pieces[8];
				rook_q = pieces[1];
				rank = 0;
				break;
			case BLACK:
				king = pieces[21];
				rook_k = pieces[24];
				rook_q = pieces[17];
				rank = 7;
				break;
		}
		if (
			not king->times_moved and
			not rook_k->times_moved and
			board[rank][5]->type == ' ' and
			board[rank][6]->type == ' ' and
			reached_by(4, rank, opponent(), true).empty() and
			reached_by(5, rank, opponent(), true).empty() and
			reached_by(6, rank, opponent(), true).empty()
		) {
			moves.insert(new Move(king, 6, rank, 'K', 0, 0, 0));
		}
		if (
			not king->times_moved and
			not rook_q->times_moved and
			board[rank][3]->type == ' ' and
			board[rank][2]->type == ' ' and
			board[rank][1]->type == ' ' and
			reached_by(4, rank, opponent(), true).empty() and
			reached_by(3, rank, opponent(), true).empty() and
			reached_by(2, rank, opponent(), true).empty()
		) {
			moves.insert(new Move(king, 2, rank, 'K', 0, 0, 0));
		}

		return moves;
	}

	void apply_move(Move* move)
	{
		auto& piece = pieces[move->index];
		piece->x = move->x_to;
		piece->y = move->y_to;
		piece->type = move->promote_to;
		piece->times_moved++;
		for (const auto& pawn : pieces) {
			pawn->enpassantable = false;
		}

		if (move->capture.index) {
			board[move->capture.y][move->capture.x]->captured = true;
			board[move->capture.y][move->capture.x] = empty;
		}

		board[move->y_to][move->x_to] = piece;
		board[move->y_from][move->x_from] = empty;

		if (piece->type == 'K') {
			Piece* rook;
			switch (move->x_to - move->x_from) {
				case 2:
					rook = board[piece->y][7];
					rook->x = 5;
					board[piece->y][5] = rook;
					board[piece->y][7] = empty;
					break;
				case -2:
					rook = board[piece->y][0];
					rook->x = 3;
					board[piece->y][3] = rook;
					board[piece->y][0] = empty;
					break;
			}
		}

		if (move->become_enpassantable)
			piece->enpassantable = true;

		to_move = opponent();
		history.push_back(move);
	}

	void undo_move(Move* move)
	{
		auto& piece = pieces[move->index];
		piece->x = move->x_from;
		piece->y = move->y_from;
		piece->type = move->promote_from;
		piece->times_moved--;
		piece->enpassantable = false;

		board[move->y_to][move->x_to] = empty;
		board[move->y_from][move->x_from] = piece;

		if (move->capture.index) {
			board[move->capture.y][move->capture.x] = pieces[move->capture.index];
			board[move->capture.y][move->capture.x]->captured = false;
		}

		if (piece->type == 'K') {
			Piece* rook;
			switch (move->x_to - move->x_from) {
				case 2:
					rook = board[piece->y][5];
					rook->x = 7;
					board[piece->y][7] = rook;
					board[piece->y][5] = empty;
					break;
				case -2:
					rook = board[piece->y][3];
					rook->x = 0;
					board[piece->y][0] = rook;
					board[piece->y][3] = empty;
					break;
			}
		}

		to_move = opponent();
		history.pop_back();

		if (not history.size())
			return;
		const auto& last_move = history.back();
		if (last_move->become_enpassantable) {
			pieces[last_move->index]->enpassantable = true;
		}
	}

	void show_board(ostream& file)
	{
		if (to_move != 1)
			return;

		file << to_move << '|';
		string delim = "";
		for (const auto& rank : board) {
			for (const auto& piece : rank) {
				file << delim;
				if (piece->id > 0)
					file << piece->id;
				delim = ",";
			}
		}
		file << "|";
		delim = "";
		for (int i=1; i<=32; ++i) {
			file << delim << pieces[i]->type;
			delim = ",";
		}
		file << "|";
		delim = "";
		for (int i=1; i<=32; ++i) {
			file << delim << (bool)pieces[i]->times_moved;
			delim = ",";
		}
		file << "|";
		for (const auto& target : pieces) {
			if (target->enpassantable) {
				file << target->id;
				break;
			}
		}
		file << "|";
		vector<vector<bool>> destinations(
				8, vector<bool>(
					8, false
		    ));

		vector<vector<int>> king_moves   = {{ 0, 1},{ 1, 1},{ 1, 0},{ 1,-1},{ 0,-1},{-1,-1},{-1, 0},{-1, 1}};
		vector<vector<int>> knight_moves = {{-2, 1},{-2,-1},{-1, 2},{-1,-2},{ 1, 2},{ 1,-2},{ 2, 1},{ 2,-1}};
		for (const auto& rank : board) {
			for (const auto& piece : rank) {
				if (piece->type == 'B' and piece->player == to_move) {
                    for (const auto& direction : king_moves) {
                        if (direction[0] == 0 or direction[1] == 0)
                            continue;
                        int _x = piece->x, _y = piece->y;
                        for(;;) {
                            _x += direction[0];
                            _y += direction[1];

                            if (_x < 0 or _x > 7 or _y < 0 or _y > 7)
                                break;

                            destinations[_y][_x] = true;
                        }
                    }
                }
			}
		}

		delim = "";
		for (const auto rank : destinations) {
			for (const auto destination : rank) {
				file << delim;
				delim = ",";
				file << destination;
			}
		}

		file << endl;
	}

	bool draw()
	{
		int i = 0;
		for (const auto& piece : pieces) {
			if (piece->captured) {
				++i;
			}
		}
		if (i == 30) {
			return true;
		}

		if (history.size() < 100)
			return false;

		i = 0;
		for (auto rit = history.rbegin(); rit != history.rend(); ++rit) {
			const auto& move = *rit;
			++i;
			if (
				move->promote_from == 'P' or
				move->capture.index != 0
			)
				return false;

			if (i == 100) {
				return true;
			}
		}
	}
};

int main()
{
	ofstream movefile;
	movefile.open("moves.csv");
	Player winner;
    int move_count = 0;
	for (int i = 0; i < 10000; ++i) {
		Board board = Board();
		set<Move*> moves;
		for(;;) {
			board.show_board(movefile);
			moves = board.legal_moves();
			if (not moves.size()) {
				if (board.in_check(WHITE))
					winner = BLACK;
				else if (board.in_check(BLACK))
					winner = WHITE;
				else
					winner = DRAW;
				break;
			}

			if (board.draw()) {
				winner = DRAW;
				break;
			}

			set<Move*>::const_iterator it(moves.begin());
			advance(it, rand()%moves.size());
			board.apply_move(*it);
            move_count++;
		}
        cout << move_count << endl;
	}

	movefile.close();

	return 0;
}
