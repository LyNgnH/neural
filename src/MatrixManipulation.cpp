#pragma once
#include "MatrixManipulation.h"
#include <iostream>
#include <random>
#include <thread>

using namespace std;


/*Helpers*/
namespace {

	const unsigned NUM_THREADS = 8;

	/* Validate if a matrix is well-formatted*/
	template <typename T>
	bool check_matrix(const matrix<T>& m) {
		if (m.empty())
			return false;
		if (m[0].empty())
			return false;
		if (m.size() < 2 && m[0].size() < 2) //single-element matrix is not allowed
			return false;
		return true;
	}

	/* Compare two matrices if they are equal in size*/
	template<typename T>
	bool compare(const matrix<T>& m1, const matrix<T>& m2) {
		unsigned rows1 = num_rows(m1);
		unsigned cols1 = num_columns(m1);
		unsigned rows2 = num_rows(m2);
		unsigned cols2 = num_columns(m2);
		if (rows1 != rows2 || cols1 != cols2)
			return false;
		return true;
	}

	/* Check if two matrices are compatible for dot product*/
	template <typename T>
	bool dotable(const matrix<T> &m1, const matrix<T> &m2) {
		if (m1.empty() || m2.empty()) //if either is empty
			return false;
		if (num_columns(m1) != num_rows(m2)) //# of rows of 1st matrix must = to # of columns of 2nd matrix
			return false;
		return true;
	}

	/* Helper for number of rows of a matrix*/
	template <typename T>
	inline unsigned num_rows(const matrix<T>& m) {
		return m.size();
	}

	/* Helper for getting number of columns of a matrix*/
	template <typename T>
	inline unsigned num_columns(const matrix<T>& m) {
		return m.empty() ? 0 : m[0].size();
	}

	/* Simple dot product algorithm*/
	template <typename T>
	void classic_dot(const matrix<T>& m1, const matrix<T>& m2, matrix<T>& result) {
		dot(m1, m2, result, 0, num_rows(m1));
	}

	/* Compute dot product using multiple threads*/
	template <typename T>
	void multithreaded_dot(const matrix<T>& m1, const matrix<T>& m2, matrix<T>& result) {
		thread threads[NUM_THREADS];
		unsigned m1_row = num_rows(m1);
		//If number of threads > number of rows of the first matrix, set thread number to number or rows
		unsigned t = m1_row > NUM_THREADS ? NUM_THREADS : m1_row; 
		//Number of rows each thread executes
		unsigned num_row_each_thread = m1_row / t;
		//Number of left-over rows
		unsigned left_over = m1_row % (num_row_each_thread * t);
		unsigned start = 0;
		unsigned end = start + num_row_each_thread;
		unsigned t_id = 0;
		//Spawn threads
		while (end <= m1_row && t_id < t) { 
			//Handle left-over rows
			threads[t_id++] = thread(dot<T>, ref(m1), ref(m2), ref(result), start, end);
			start = end;
			end += num_row_each_thread;
		}
		//Handle left-over rows
		dot(m1, m2, result, m1_row - left_over, m1_row);
		for (unsigned i = 0; i < t_id; i++) {
			threads[i].join();
		}
	}

	/* Cache-friendly dot product*/
	template<typename T>
	void dot(const matrix<T>& m1, const matrix<T>& m2, matrix<T>& res, unsigned start, unsigned end) {
		unsigned m2_cols = num_columns(m2);
		unsigned m2_rows = num_rows(m2);
		vector<T> m2_tmp(m2_rows);
		for (unsigned j = 0; j < m2_cols; j++)
		{
			for (unsigned k = 0; k < m2_rows; k++)
				m2_tmp[k] = m2[k][j];

			for (unsigned i = start; i < end; i++)
			{
				for (unsigned k = 0; k < m2_rows; k++)
					res[i][j] += m1[i][k] * m2_tmp[k];
			}
		}
	}

}

/*Custom Exceptions*/
class Matrix_Formatting_Error : public exception {
public:
	const char * what() const throw() {
		return "ERROR: Matrix is not well formatted\n";
	}
};
class Out_Of_Range_Error : public exception {
public:
	const char * what() const throw() {
		return "ERROR: Index out of range.\n";
	}
};
class Incompatible_Matrix : public exception {
public:
	const char * what() const throw() {
		return "ERROR: Input matrix is not compatible\n";
	}
};

							/*Class definition*/
//Constructor:
template<typename T>
MatrixManipulation<T>::MatrixManipulation() {}

template<typename T>
MatrixManipulation<T>::MatrixManipulation(matrix<T>& m)
	: content(m) 
{
    if (!check_matrix(m)) {
        Matrix_Formatting_Error err;
        throw err;
    }
}

template<typename T>
MatrixManipulation<T>::MatrixManipulation(unsigned row, unsigned col)
	: content(row, vector<T>(col, 0))
{
	if (!check_matrix(content)) {
		Matrix_Formatting_Error err;
		throw err;
	}
}

/* Fill this matrix with random number. Starts at lower_lim and end at upper_lim*/
template<typename T>
void MatrixManipulation<T>::fill(T lower_lim, T upper_lim) {
	//Random number generator
	default_random_engine generator;
	uniform_real_distribution<T> distribution(lower_lim, upper_lim);

	//Fill matrix
	for (unsigned i = 0; i < num_rows(content); i++) {
		for (unsigned j = 0; j < num_columns(content); j++) {
			content[i][j] = distribution(generator);
		}
	}
}

//Modifiers:
template<typename T>
void MatrixManipulation<T>::set(unsigned row_i, unsigned col_i, T val) {
	if (row_i >= num_rows(content) || col_i >= num_columns(content)) {
		Out_Of_Range_Error err;
		throw err;
	}
	content[row_i][col_i] = val;
}

//Compute and return dot product
template <typename T>
MatrixManipulation<T> MatrixManipulation<T>::dot_product(const MatrixManipulation<T> &other) const {
	//Check compatibility
    if (dotable(content, other.content)) {
		int dot_rows = num_rows(content);
		int dot_cols = num_columns(other.content);
        matrix<T> ret(dot_rows, vector<T>(dot_cols, 0));
		//classic_dot(content, other.content, ret);
		multithreaded_dot(content, other.content, ret);
		return MatrixManipulation<T>(ret);
    }
    else {
        Incompatible_Matrix err;
        throw err;
    }
}

//Overloading operators:
template<typename T>
MatrixManipulation<T> MatrixManipulation<T>::operator+(const MatrixManipulation<T>& other) const {
	return dot_product(other);
}

template<typename T>
MatrixManipulation<T> MatrixManipulation<T>::operator-(const MatrixManipulation<T>& other) const {
	if (!compare(content, other.content)) {
		Incompatible_Matrix err;
		throw err;
	}
	unsigned rows = num_rows(content);
	unsigned cols = num_columns(content);
	MatrixManipulation<T> ret(rows, cols);
	for (unsigned i = 0; i < rows; i++) {
		for (unsigned j = 0; j < cols; j++) {
			T val = content[i][j] - other.content[i][j];
			ret.set(i, j, val);
		}
	}
	return ret;
}


//Create a transposed copy of this matrix
template <typename T>
MatrixManipulation<T> MatrixManipulation<T>::transpose() const {
	int new_rows = num_columns(content);
	int new_cols = num_rows(content);
	matrix<T> res(new_rows, vector<T>(new_cols, 1));
	for (unsigned i = 0; i < res.size(); i++) {
		for (unsigned j = 0; j < res[0].size(); j++) {
			res[i][j] = content[j][i];
		}
	}
	return MatrixManipulation<T>(res);
}


//Getters:
template <typename T>
T MatrixManipulation<T>::get(unsigned row, unsigned col) const {
	if (row >= num_rows(content) || col >= num_columns(content)) {
		Out_Of_Range_Error err;
		throw err;
	}
	return content[row][col];
}

template <typename T>
unsigned MatrixManipulation<T>::get_num_rows() {
	return num_rows(content);
}

template <typename T>
unsigned MatrixManipulation<T>::get_num_columns() {
	return num_columns(content);
}


//Display matrix to console
template <typename T>
void MatrixManipulation<T>::print() {
	for (unsigned i = 0; i < content.size(); i++) {
		for (unsigned j = 0; j < content[0].size(); j++) {
			if (j == content[0].size() - 1)
				std::cout << content[i][j];
			else
				std::cout << content[i][j] << "\t";
		}
		cout << "\n\n";
	}
}

template <typename T>
MatrixManipulation<T>::~MatrixManipulation() {
}