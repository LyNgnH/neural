
#include "NeuralNetwork.cpp"
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace chrono;

template<typename T>
unsigned train_n_test(const string& file_name, NeuralNetwork<T>& nw, bool is_training) {
	unsigned counter = 0;
	ifstream infile(file_name);
	string delimiter = ",";

	//Parse file
	while(infile) { 
		string s;

		//Get line
		if (!getline(infile, s)) break; 
		vector<T> record;

		//Get element separated by commas
		unsigned n = s.find(delimiter); 
		int label = stoi(s.substr(0, n));
		s.erase(0, n + delimiter.length());
		while ((n = s.find(delimiter)) != std::string::npos) {
			string token = s.substr(0, n);
			s.erase(0, n + delimiter.length());
			T val = stoi(token);
			T normalized = (val / 255) * 0.99 + 0.01;
			record.push_back(normalized);
		}
		record.push_back((stoi(s) / 255 * 0.99) + 0.1);
		vector<T> target(10, 0.1);
		target[label] = 0.99;
		
		//Start training
		if (is_training) {
			nw.train(record, target);
			cout << "Has trained: " << ++counter << " items" << endl;
		}
		//If training flag is off, do testing 
		else {

			//Make a query to the network
			MatrixManipulation<T> result = nw.query(record);
			T greatest = result.get(0, 0);
			int index = 0;

			//Find the most likely output i.e. the greatest element in the vector
			for (unsigned i = 1; i < result.get_num_rows(); i++) {
				if (result.get(i, 0) > greatest) {
					greatest = result.get(i, 0);
					index = i;
				}
			}

			//Print results, compare it to target
			cout << endl;
			cout << "Target = " << label << "\t|\t" << "Result = " << index << endl;
			if (index == label) {
				cout << "Correct!" << endl;
				counter++;
			}
			else
				cout << "Incorrect!" << endl;
		}
	}
	if (!infile.eof()) {
		cerr << "End of file\n";
	}
	return counter;
}

void init() {

	//Init
	NeuralNetwork<double> neural_net(784, 100, 10, 0.3);
	neural_net.initialize();

	//Get user input
	string f;
	cout << "Enter name for training file: ";
	cin >> f;
	string f1;
	cout << "Enter name for test file: ";
	cin >> f1;

	//Start training
	cout << "Training..." << endl;
	auto start = chrono::high_resolution_clock::now();
	auto total = train_n_test(f, neural_net, true); //Number of tests
	auto end = chrono::high_resolution_clock::now();
	auto duration1 = std::chrono::duration_cast<microseconds> (end - start);
	cout << "Finish training." << endl;

	//Start testing
	cout << "Testing..." << endl;
	start = chrono::high_resolution_clock::now();
	auto passes = train_n_test(f1, neural_net, false); //Number of passes
	end = chrono::high_resolution_clock::now();
	auto duration2 = std::chrono::duration_cast<microseconds> (end - start);
	cout << "Finish testing." << endl;

	//Print results
	cout << endl;
	cout << "Total training time:\t\t" << duration1.count() << endl;
	cout << "Total testing time:\t\t" << duration2.count() << endl;
	cout << "Number of tests:\t\t" << total << endl;
	cout << "Number of passes:\t\t" << passes << endl;
	cout << "Accuracy:\t\t\t" << ((float)passes / (float)total) * 100 << "%" << endl;
}

int main() {
	//Run example
	init();
};