
template <typename T>
bool dlmread(const char *filename, int &numof_rows, int &numof_cols, vector<T> &arr)
{
	// try to open the input file
	ifstream InputFile;
	InputFile.open(filename);
	if(!InputFile.is_open()) {
		std::cout<<"cannot open file "<<filename<<std::endl;
		return false;
	}
	// determine number of rows and cols
	numof_rows = 1;
	numof_cols = 0;
	string strLine, strele;
	getline(InputFile, strLine);
	istringstream szline(strLine);
	while( szline >> strele ) {
		numof_cols++;
	}
	while( getline(InputFile, strLine) ) {
		numof_rows++;
	}
	InputFile.close();
	
	// read data
	arr.resize(numof_cols*numof_rows);
	InputFile.open(filename);
	for(int i = 0; i < numof_cols*numof_rows; i++)
		InputFile >> arr[i];
	InputFile.close();
	
	return true;
}


template <typename T>
bool dlmwrite(const char *filename, int numof_rows, int numof_cols, const vector<T>& arr)
{
	ofstream OutputFile(filename);
	if(!OutputFile.is_open()) {
		std::cerr<<"cannot open file "<<filename<<std::endl;
		return false;
	}
	for(int i = 0; i < numof_rows; i++) {
		for(int j = 0; j < numof_cols; j++) {
			OutputFile << arr[i*numof_cols+j]
					   << ((j < numof_cols - 1) ? " " : "\n");
		}
	}
	OutputFile.close();
	return true;
}


template <typename T>
bool dlmwrite(const char *filename, int numof_rows, int numof_cols, const T *arr)
{
	ofstream OutputFile(filename);
	if(!OutputFile.is_open()) {
		std::cerr<<"cannot open file "<<filename<<std::endl;
		return false;
	}
	for(int i = 0; i < numof_rows; i++) {
		for(int j = 0; j < numof_cols; j++) {
			OutputFile << arr[i*numof_cols+j]
					   << ((j < numof_cols - 1) ? " " : "\n");
		}
	}
	OutputFile.close();
	return true;
}




template <typename T>
bool mpi_dlmwrite(const char *filename, const T *pArr, const int &numof_rows, const int &numof_cols, MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);
	
	if(0 == rank) remove(filename);
	MPI_Barrier(comm);
	
	ofstream output;
	for(int k = 0; k < nproc; k++) {
		if(rank == k) {
			output.open(filename, ios::app|ios::out);
			for(int i = 0; i < numof_rows; i++) {
				for(int j = 0; j < numof_cols; j++) 
					output<<pArr[i*numof_cols+j]<<" ";
				output<<endl;
			}
			output.flush();
			output.close();
		}
		MPI_Barrier(comm);
	}

	return true;
}


