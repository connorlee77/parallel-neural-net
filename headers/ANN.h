#ifndef ANN_H
#define ANN_H

/*******************CHANGE PATH HERE! ***************************************/
std::string PATH = "C:\\Users\\conno\\Desktop\\neural-net-parallel\\rawdata\\";

/*****************************************************************************/





std::string trainImages = "train-images.idx3-ubyte";
std::string trainLabels = "train-labels.idx1-ubyte";
std::string testImages = "t10k-images.idx3-ubyte";
std::string testLabels = "t10k-labels.idx1-ubyte";

class ANN {

public:

	int *shape;
	int layer_count;
	int sum_weights;
	int bias_size;

	float *weights;
	float *bias;
	int *weight_index;
	int *bias_index;
	int * io_index;
	
	float *deltas;
	float *input_layers;
	float *output_layers;
	float *input_host;
	float *labels;

	int *weight_shapes;
	int network_size;
	int normal_size;

	float *delta_w;
	float *delta_b;

	float *eval_input;
	float *eval_output;
	float *host_weights;
	float *host_bias;
	int *normal_io_index;
	
	/**
	 * @param shape					array 	[input, hidden ... , output]
	 * @param layer_count			int 	layer count
	 * @param cost_function 		string	"quadratic, entropy"
	 * @param activation_function	string 	"ReLu, softmax, tanh"
	 * @param regularization		int 	"0 - None, 1 - L1, 2 - L2"
	 */
	ANN(int *shape, int layer_count);

	~ANN();
			
	void ANN::initWeights(int layer_count);
	
	float *feedforward(int data_idx, cudaStream_t s);
	float *cpu_feedforward(float *input_data, int size);
	void backpropogate(float label, int data_idx, cudaStream_t s);

	void sgd(float **training_data, float* training_labels, float **testing_data, float* testing_labels, int size, float gamma, float alpha, int epochs, int input_data_size, int test_size);

	int evaluate(float **testing_data, float *testing_labels, int size, int input_data_size);
};

#endif // ANN_H