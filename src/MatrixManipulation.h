#pragma once
#include <vector>

using namespace std;

template <typename T>
using matrix = vector<vector<T>>;

/**
Matrix manipulator. This class provides
some common matrix operations
**/
template <typename T>
class MatrixManipulation
{
	
private:
	/* 2d vector instance */
	matrix<T> content;

public:
	/* constructor */
	MatrixManipulation();
	MatrixManipulation(matrix<T>& m);
	MatrixManipulation(unsigned row, unsigned col); //Create empty matrix given the matrix's dimension

	/* fill matrix with generated random elements */
	void fill(T lower_lim, T upper_lim);

	/* compute dot product */
	MatrixManipulation<T> dot_product(const MatrixManipulation<T>& other) const;

	/*overloaded operators*/
	MatrixManipulation<T> operator+(const MatrixManipulation<T>& other) const;
	MatrixManipulation<T> operator-(const MatrixManipulation<T>& other) const;

    /* Transpose matrix */
	MatrixManipulation<T> transpose() const;

	/* get a single element */
	T get(unsigned row, unsigned col) const;

	/* set value at */
	void set(unsigned row_i, unsigned col_i, T val);
	
	/* print matrix */
	void print();

	/*get dimesions*/
	unsigned get_num_rows();
	unsigned get_num_columns();

	/* destructor */
	~MatrixManipulation();
};



