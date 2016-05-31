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

	float *weights;
	float *bias;

	float *delta_weights;
	float *delta_bias;
	float *deltas;
	float *input_layers;
	float *output_layers;

	/**
	 * @param shape					array 	[input, hidden ... , output]
	 * @param layer_count			int 	layer count
	 * @param cost_function 		string	"quadratic, entropy"
	 * @param activation_function	string 	"ReLu, softmax, tanh"
	 * @param regularization		int 	"0 - None, 1 - L1, 2 - L2"
	 */
	ANN(int *shape, int layer_count);

	~ANN();
		
	void initLayers(float **input_layers, float **output_layers, float **deltas, int layer_count, int *shape);
	
	void ANN::initWeights(float **weights, float **bias, float **delta_w, float **delta_b, int layer_count, int *shape);
	
	float *feedforward(float *array, int size);
	void backpropogate(float label);

	void sgd(float **training_data, float* training_labels, float **testing_data, float* testing_labels, int size, float gamma, float alpha, int epochs, int input_data_size, int test_size);

	int ANN::evaluate(float **testing_data, float *testing_labels, int size, int input_data_size);
};

#endif // ANN_H