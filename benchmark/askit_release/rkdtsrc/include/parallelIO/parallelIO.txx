
template <typename T>
bool dlmread(const char *filename, int &numof_points, int &dim, vector<T> &arr, bool flag_read_all)
{
	// try to open the input file
	ifstream InputFile;
	InputFile.open(filename);
    if(!InputFile.is_open()) {
		std::cout<<"cannot open file "<<filename<<std::endl;
		return false;
	}
	// determine number of rows and cols
	int glb_numof_points = 1;
	dim = 0;
	string strLine, strele;
	getline(InputFile, strLine);
	istringstream szline(strLine);
	while( szline >> strele ) dim++;
	while( getline(InputFile, strLine) ) glb_numof_points++;
	InputFile.close();

    if(flag_read_all == false) {
        if(numof_points > glb_numof_points) {
            cout<<"warning: no enough data in "<<filename
                <<", read all available "<<glb_numof_points<<" points"<<endl;
            numof_points = glb_numof_points;
        }
    } else {
        numof_points = glb_numof_points;
    }

	// read data
	arr.resize(dim*(long)numof_points);
	InputFile.open(filename);
	for(long i = 0; i < dim*(long)numof_points; i++)
		InputFile >> arr[i];
	InputFile.close();

	return true;
}

template <typename T>
bool dlmread(const char *filename, int &numof_points, int &dim, T *&arr, bool flag_read_all)
{
	// try to open the input file
	ifstream InputFile;
	InputFile.open(filename);
    if(!InputFile.is_open()) {
		std::cout<<"cannot open file "<<filename<<std::endl;
		return false;
	}
	// determine number of rows and cols
	int glb_numof_points = 1;
	dim = 0;
	string strLine, strele;
	getline(InputFile, strLine);
	istringstream szline(strLine);
	while( szline >> strele ) dim++;
	while( getline(InputFile, strLine) ) glb_numof_points++;
	InputFile.close();

    if(flag_read_all == false) {
        if(numof_points > glb_numof_points) {
            cout<<"warning: no enough data in "<<filename
                <<", read all available "<<glb_numof_points<<" points"<<endl;
            numof_points = glb_numof_points;
        }
    } else {
        numof_points = glb_numof_points;
    }

	// read data
    arr = new T [(long)numof_points*dim];
	InputFile.open(filename);
	for(long i = 0; i < dim*(long)numof_points; i++)
		InputFile >> arr[i];
	InputFile.close();

	return true;
}



template <typename T>
bool mpi_dlmread(const char *filename, long &numof_points, int &dim,
                 vector<T> &arr, MPI_Comm comm, bool flag_read_all)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

	// try to open the input file
	ifstream InputFile;
	InputFile.open(filename);
    if(!InputFile.is_open()) {
		std::cout<<"cannot open file "<<filename<<std::endl;
		return false;
	}

	// determine number of rows and cols
	long glb_numof_points = 1;
	dim = 0;
	string strLine, strele;
    if(rank == 0) {
	    getline(InputFile, strLine);
	    istringstream szline(strLine);
	    while( szline >> strele ) dim++;
	    while( getline(InputFile, strLine) ) glb_numof_points++;
    }
	InputFile.close();
    MPI_Bcast(&glb_numof_points, 1, MPI_LONG, 0, comm);
    MPI_Bcast(&dim, 1, MPI_INT, 0, comm);

    // check whether read all of data
    if(flag_read_all == false) {
        if(numof_points > glb_numof_points) {
            cout<<"warning: no enough data ("<<numof_points<<") in "<<filename
                <<", read all available "<<glb_numof_points<<" points"<<endl;
        } else {
            glb_numof_points = numof_points;
        }
    }

    // calculate the numof_points read in each processor
    long divd = glb_numof_points / size;
    long rem = glb_numof_points % size;
    numof_points = rank < rem ? (divd+1) : divd;

    //long mppn = glb_numof_points / size;
    //if(rank != size-1) {
    //    numof_points = mppn;
    //} else {
    //    numof_points = glb_numof_points - mppn*rank;
    //}

    // read data
	arr.resize(dim*numof_points);
    long offset = 0;
    long dummy_numof_points = numof_points;
    MPI_Scan(&dummy_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
    offset -= dummy_numof_points;
    // skip lines to its own processor's line
	InputFile.open(filename);
    long nline = 0;
    while( nline < offset ) {
        getline(InputFile, strLine);
        nline++;
    }
    long pn = 0;
    while( getline(InputFile, strLine) && pn < numof_points ) {
	    istringstream szline(strLine);
        int pdim = 0;
        while(szline >> arr[pn*dim+pdim]) {
            pdim++;
        }
        pn++;
    }
	InputFile.close();

    MPI_Barrier(comm);

	return true;
}

template <typename T>
bool mpi_dlmread(const char *filename, long &numof_points, int &dim,
                 T *&arr, MPI_Comm comm, bool flag_read_all)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

	// try to open the input file
	ifstream InputFile;
	InputFile.open(filename);
    if(!InputFile.is_open()) {
		std::cout<<"cannot open file "<<filename<<std::endl;
		return false;
	}

	// determine number of rows and cols
	long glb_numof_points = 1;
	dim = 0;
	string strLine, strele;
    if(rank == 0) {
	    getline(InputFile, strLine);
	    istringstream szline(strLine);
	    while( szline >> strele ) dim++;
	    while( getline(InputFile, strLine) ) glb_numof_points++;
    }
	InputFile.close();
    MPI_Bcast(&glb_numof_points, 1, MPI_LONG, 0, comm);
    MPI_Bcast(&dim, 1, MPI_INT, 0, comm);

    // check whether read all of data
    if(flag_read_all == false) {
        if(numof_points > glb_numof_points) {
            cout<<"warning: no enough data ("<<numof_points<<") in "<<filename
                <<", read all available "<<glb_numof_points<<" points"<<endl;
        } else {
            glb_numof_points = numof_points;
        }
    }

    // calculate the numof_points read in each processor
    long divd = glb_numof_points / size;
    long rem = glb_numof_points % size;
    numof_points = rank < rem ? (divd+1) : divd;

    //long mppn = glb_numof_points / size;
    //if(rank != size-1) {
    //    numof_points = mppn;
    //} else {
    //    numof_points = glb_numof_points - rank*mppn;
    //}


    // read data
    arr = new T [numof_points*dim];
    int offset = 0;
    MPI_Scan(&numof_points, &offset, 1, MPI_INT, MPI_SUM, comm);
    offset -= numof_points;
    // skip lines to its own processor's line
	InputFile.open(filename);
    int nline = 0;
    while( nline < offset ) {
        getline(InputFile, strLine);
        nline++;
    }
    long pn = 0;
    while( getline(InputFile, strLine) && pn < numof_points ) {
	    istringstream szline(strLine);
        int pdim = 0;
        while(szline >> arr[pn*dim+pdim]) {
            pdim++;
        }
        pn++;
    }
	InputFile.close();

    MPI_Barrier(comm);

	return true;
}




template <typename T>
bool binread(const char *filename, int numof_points, int dim, vector<T> &points)
{
    points.resize(numof_points*dim);

    ifstream infile;
    infile.open(filename, ifstream::binary);
    infile.read((char*)(&(points[0])), (long)numof_points*dim*sizeof(points[0]));
    if(infile.gcount() != (long)numof_points*dim*sizeof(points[0])) {
        cout<<"warning: corrupted read detected: insufficient data in file "<<filename<<endl;
        return false;
    }
    infile.close();

    return true;
}

template <typename T>
bool binread(const char *filename, int numof_points, int dim, T *&points)
{
    points = new double [numof_points*dim];

    ifstream infile;
    infile.open(filename, ifstream::binary);
    infile.read((char*)(&(points[0])), (long)numof_points*dim*sizeof(points[0]));
    if(infile.gcount() != (long)numof_points*dim*sizeof(points[0])) {
        cout<<"warning: corrupted read detected: insufficient data in file "<<filename<<endl;
        return false;
    }
    infile.close();

    return true;
}




template <typename T>
bool mpi_binread(const char *filename, long glb_numof_points, int dim,
                    int &numof_points, vector<T> &points, MPI_Comm comm)
{
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    long divd = glb_numof_points / size;
    long rem = glb_numof_points % size;
    numof_points = rank < rem ? (divd+1) : divd;

    //long mppn = glb_numof_points / size;
    //if(rank != size-1) {
    //    numof_points = mppn;
    //} else {
    //    numof_points = glb_numof_points - rank*mppn;
    //}

    long offset = 0;
    long dummy_numof_points = numof_points;
    MPI_Scan(&dummy_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
    offset -= dummy_numof_points;
    points.resize((long)numof_points*(long)dim);
    long byte_offset = offset*(long)dim*(long)sizeof(points[0]);

    ifstream infile;
    infile.open(filename, ifstream::binary);
    //infile.seekg(byte_offset, ifstream::beg);
    infile.seekg(byte_offset, ios_base::beg);
    infile.read((char*)(&(points[0])), numof_points*dim*sizeof(points[0]));
    if(infile.gcount() != numof_points*dim*sizeof(points[0])) {
        cout<<"warning: rank "<<rank<<" corrupted read detected: insufficient data in file "<<filename
            <<" infile.gcount() = "<<infile.gcount()
            <<", numof_points*dim*sizeof(double) = "<<numof_points*dim*sizeof(points[0])<<endl;
        infile.close();
        return false;
    }
    infile.close();

    return true;
}

template <typename T>
bool mpi_binread(const char *filename, long glb_numof_points, int dim,
                    int &numof_points, T *&points, MPI_Comm comm)
{
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    long divd = glb_numof_points / size;
    long rem = glb_numof_points % size;
    numof_points = rank < rem ? (divd+1) : divd;

    //long mppn = glb_numof_points / size;
    //if(rank != size-1) {
    //    numof_points = mppn;
    //} else {
    //    numof_points = glb_numof_points - rank*mppn;
    //}

    long offset = 0;
    long dummy_numof_points = numof_points;
    MPI_Scan(&dummy_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
    offset -= dummy_numof_points;
    points = new T [(long)numof_points*(long)dim];
    long byte_offset = offset*(long)dim*(long)sizeof(points[0]);

    ifstream infile;
    infile.open(filename, ifstream::binary);
    infile.seekg(byte_offset, ifstream::beg);
    infile.read((char*)(&(points[0])), (long)numof_points*(long)dim*(long)sizeof(points[0]));
    if(infile.gcount() != numof_points*dim*sizeof(points[0])) {
        cout<<"warning: rank "<<rank<<" corrupted read detected: insufficient data in file "<<filename
            <<" infile.gcount() = "<<infile.gcount()
            <<", numof_points = "<<numof_points
            <<", dim = "<<dim
            <<", sizeof(points[0]) = "<<sizeof(points[0])
            <<", sizeof(double) = "<<sizeof(double)
            <<", numof_points*dim*sizeof(points[0]) = "<<numof_points*dim*sizeof(points[0])
            <<", numof_points*dim*sizeof(double) = "<<numof_points*dim*sizeof(double)
            <<endl;
        infile.close();
        return false;
    }
    infile.close();

    return true;
}



template <typename T>
bool dlmwrite(const char *filename, int numof_points, int dim, const T *arr)
{
	ofstream OutputFile(filename);
	if(!OutputFile.is_open()) {
		std::cerr<<"cannot open file "<<filename<<std::endl;
		return false;
	}
	for(long i = 0; i < numof_points; i++) {
		for(long j = 0; j < dim; j++) {
			OutputFile << arr[i*dim+j]
					   << ((j < dim - 1) ? " " : "\n");
		}
	}
	OutputFile.close();
	return true;
}


template <typename T>
bool binwrite(const char *filename, int numof_points, int dim, const T *points)
{
    if(is_file_exist(filename)) remove(filename);

    ofstream outfile;
    outfile.open(filename, ios::binary|ios::app);
    if(!outfile.is_open()) {
        cout<<"cannot open "<<filename<<endl;
        return false;
    }
    outfile.write((char*)points, (long)numof_points*(long)dim*(long)sizeof(points[0]));
    outfile.close();
    return true;
}


template <typename T>
bool mpi_dlmwrite(const char *filename, int numof_points, int dim, const T *arr, MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

	if(0 == rank) remove(filename);
	MPI_Barrier(comm);

	ofstream output;
    for(int r = 0; r < nproc; r++) {
        if(rank == r) {
			output.open(filename, ios::app|ios::out);
            for(long i = 0; i < numof_points; i++) {
				for(long j = 0; j < dim; j++)
					output<<arr[i*(long)dim+j]<<" ";
				output<<endl;
			}
			output.flush();
			output.close();
		}
		MPI_Barrier(comm);
	}

	return true;
}


template <typename T>
bool mpi_binwrite(const char *filename, int numof_points, int dim, const T *arr, MPI_Comm comm)
{
	int rank, nproc;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nproc);

    double start_t = omp_get_wtime();

    if(0 == rank) remove(filename);
	MPI_Barrier(comm);

    long dummy_numof_points = numof_points;
    long offset;
    MPI_Scan(&dummy_numof_points, &offset, 1, MPI_LONG, MPI_SUM, comm);
    offset -= numof_points;
    long byte_offset = offset*(long)dim*(long)sizeof(arr[0]);

	ofstream output;
    for(int r = 0; r < nproc; r++) {
        if(rank == r) {
            //cout<<"rank "<<rank<<": write "<<numof_points<<" points, \tpoint_offset = "<<offset<<", \tbyte_offset = "<<byte_offset<<" \t... ";
			output.open(filename, ios::app|ios::binary);
            output.seekp(byte_offset, ios_base::beg);
            output.write((char*)arr, (long)numof_points*(long)dim*(long)sizeof(arr[0]));
			output.close();
            //cout<<" done! "<<omp_get_wtime()-start_t<<endl;
		}
		MPI_Barrier(comm);
	}

	return true;
}







