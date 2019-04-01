//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Aashish Yadavally, Akash Saurabh
 *  @version 1.0
 *  @date    March 23, 2019
 *  @see     LICENSE (MIT style license file)
 */

package Perceptron
import scalation.columnar_db.Relation
import scala.math.{exp, log, sqrt}
import scalation.math.{FunctionS2S, sq}
import scalation.analytics.ActivationFun._
import scalation.util.banner
import scala.collection.mutable.Set
import scalation.linalgebra._
import scalation.analytics._
import scalation.plot.PlotM
import RegTechnique._


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Defining IllegalChoiceException class
*/
class IllegalChoiceException(s: String) extends Exception(s){}

class Exception1{
	@throws(classOf[IllegalChoiceException])
	def validate(choice: Int){
		if((choice < 0) || (choice > 11)) {
			throw new IllegalChoiceException("Invalid Choice.")
		}
	}
}


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The 'Project2' object uses the pre-defined MatrixD, Regression, Perceptron
* NeuralNet_3L and NeuralNet_XL classes to perform regression and subsequent analysis 
* on different numerical datasets, in the 'data' folder.  
*  > "sbt run" in the Scalation folder containing the build file to run the program.
* User gets two choices, once, to run on the dataset of his/her choice and again, to 
* choose the model to build the R2-RBar2-RCV2 graph on.
*/			
object Project2 extends App {

	def tran_regression(x: MatriD, y: VectoD, transform_function: FunctionS2S, transform_inverse: FunctionS2S){
		banner ("Implementing Transformed Regression... ")
		val	tran_reg = new TranRegression (x, y, null, transform_function, transform_inverse)
		val fs_cols = Set(0)				// Selected features 
		val RSqNormal = new VectorD (x.dim2)
		val RSqAdj = new VectorD (x.dim2) 
		val RSqCV = new VectorD (x.dim2)
		val n = VectorD.range(1, x.dim2)

		for (j <- 1 until x.dim2){
			val (add_var, new_param, new_qof) = tran_reg.forwardSel(fs_cols, false)
			if (add_var != -1) {
				fs_cols += add_var
				RSqNormal(j) = 100 * new_qof(0)	
				RSqAdj(j) = 100 * new_qof(7)
				val x_cv = x.selectCols(fs_cols.toArray)	// Obtaining X-matrix for selected features
				val tran_reg_cv = new TranRegression(x_cv, y, null, transform_function, transform_inverse)
				val cv_result = tran_reg_cv.crossVal()
				RSqCV(j) = 100 * cv_result(tran_reg_cv.index_rSq).mean
			}		
		}	
		val plot_mat = new MatrixD (3, x.dim2-1)
		plot_mat.update(0, RSqAdj(1 until x.dim2))
		plot_mat.update(1, RSqNormal(1 until x.dim2))
		plot_mat.update(2, RSqCV(1 until x.dim2))
		new PlotM(n, plot_mat, lines=true).saveImage("tran_regression.png")
		banner ("Successfully implemented Transformed Regression!")
	}

		
	def perceptron(x: MatriD, y: VectoD, activation_f1: AFF, hp: HyperParameter ){
		banner ("Implementing Perceptron... ")
		val	percep = Perceptron (x :^+ y, null, hp, activation_f1)
		percep.train ().eval ()
		val fs_cols = Set(0)				// Selected features 
		val RSqNormal = new VectorD (x.dim2)
		val RSqAdj = new VectorD (x.dim2) 
		val RSqCV = new VectorD (x.dim2)
		val n = VectorD.range(1, x.dim2)

		for (j <- 1 until x.dim2){
			val (add_var, new_param, new_qof) = percep.forwardSel(fs_cols, false)
			if (add_var != -1) {
				fs_cols += add_var
				val x_cv = x.selectCols(fs_cols.toArray)	// Obtaining X-matrix for selected features
			//	val nn_j   = Perceptron (x_cols :^+ y, f1 = f_)
				val percep_cv = Perceptron(x_cv :^+ y, null, hp, activation_f1)
				val cv_result = percep_cv.crossVal()
				RSqNormal(j) = 100 * new_qof(percep_cv.index_rSq)
				RSqAdj(j) = 100 * new_qof(percep_cv.index_rSqBar)
				RSqCV(j) = 100 * cv_result(percep_cv.index_rSq).mean
				println(RSqNormal(j), RSqAdj(j), RSqCV(j))
			}
		}
		val plot_mat = new MatrixD (3, x.dim2-1)
		plot_mat.update(0, RSqAdj(1 until x.dim2))
		plot_mat.update(1, RSqNormal(1 until x.dim2))
		plot_mat.update(2, RSqCV(1 until x.dim2))
		new PlotM(n, plot_mat, lines=true).saveImage("Perceptron.png")
		banner ("Successfully implemented Perceptron!")
	
	}

	def neuralnet_3l(x: MatriD, y:VectoD, activation_f0: AFF, hp: HyperParameter) {
		banner("Implementing NeuralNet3L...")
		val nn_3l = NeuralNet_3L(x :^+ y, null, -1, hp, activation_f0, f_lreLU)
		nn_3l.train ().eval ()
		println(nn_3l.report)
		val fit = nn_3l.fitA(0)
		val fs_cols = Set(0)				// Selected features 
		val RSqNormal = new VectorD (x.dim2)
		val RSqAdj = new VectorD (x.dim2) 
		val RSqCV = new VectorD (x.dim2)
		val n = VectorD.range(1, x.dim2)

		for (j <- 1 until x.dim2){
			val (add_var, new_param, new_qof) = nn_3l.forwardSel(fs_cols, false)
			if (add_var != -1) {
				fs_cols += add_var
				val x_cv = x.selectCols(fs_cols.toArray)	// Obtaining X-matrix for selected features
				val nn_3l_cv = NeuralNet_3L(x_cv :^+ y, null, -1, hp, activation_f0, f_lreLU)
				val cv_result = nn_3l_cv.crossVal()
				RSqNormal(j) = 100 * new_qof(fit.index_rSq)
				RSqAdj(j) = 100 * new_qof(fit.index_rSqBar)
				RSqCV(j) = 100 * cv_result(fit.index_rSq).mean
			}
		}
		val plot_mat = new MatrixD (3, x.dim2-1)
		plot_mat.update(0, RSqAdj(1 until x.dim2))
		plot_mat.update(1, RSqNormal(1 until x.dim2))
		plot_mat.update(2, RSqCV(1 until x.dim2))
		new PlotM(n, plot_mat, lines=true).saveImage("neuralnet_3l.png")
		banner ("Successfully implemented NeuralNet_3L!")
	}

	def neuralnet_xl(x: MatriD, y: VectoD, activation_f: Array [AFF], hp: HyperParameter) {
		banner("Implementing NeuralNetXL...")
		val nn_xl = NeuralNet_XL(x :^+ y, null, null, hp, activation_f)
		nn_xl.train ().eval ()
		val fit = nn_xl.fitA(0)
		val fs_cols = Set(0)				// Selected features 
		val RSqNormal = new VectorD (x.dim2)
		val RSqAdj = new VectorD (x.dim2) 
		val RSqCV = new VectorD (x.dim2)
		val n = VectorD.range(1, x.dim2)

		for (j <- 1 until x.dim2){
			val (add_var, new_param, new_qof) = nn_xl.forwardSel(fs_cols, false)
			if (add_var != -1) {
				fs_cols += add_var
				val x_cv = x.selectCols(fs_cols.toArray)	// Obtaining X-matrix for selected features
				val nn_xl_cv = NeuralNet_XL(x_cv :^+ y, null, null, hp, activation_f)
				val cv_result = nn_xl_cv.crossVal()
				RSqNormal(j) = 100 * new_qof(fit.index_rSq)
				RSqAdj(j) = 100 * new_qof(fit.index_rSqBar)
				RSqCV(j) = 100 * cv_result(fit.index_rSq).mean
			}
		}
		val plot_mat = new MatrixD (3, x.dim2-1)
		plot_mat.update(0, RSqAdj(1 until x.dim2))
		plot_mat.update(1, RSqNormal(1 until x.dim2))
		plot_mat.update(2, RSqCV(1 until x.dim2))
		new PlotM(n, plot_mat, lines=true).saveImage("neuralnet_xl.png")
		banner ("Successfully implemented NeuralNet_XL!")
	}
	
	def main(){
		// Giving user the choice to select from ten datasets, or to give data path to own CSV file
		println("-"*75)
		println (" Select dataset: \n\t 1. Auto MPG \n\t 2. CSM Dataset \n\t 3. Concrete Compressive Strength Dataset \n\t 4. Slump Dataset \n\t 5. Energy Dataset \n\t 6. Computer Hardware")
		println("\t 7. Airfoil Prediction Dataset \n\t 8. Combined Cycle Powerplant \n\t 9. Yacht Dataset \n\t 10. Naval Propulsion Dataset \n\t 11. For other datasets, enter: /correct/path/to/data/csv")
		println("-"*75)
		
		val choice	 = scala.io.StdIn.readLine()
		// Exception, to alert if user choice is not between 1 and 11
		var e = new Exception1()
		try {
			e.validate(choice.toInt)
		} catch {
			case ex: Exception => println("Exception Occured : " + ex)
		}
						
		val filename = if(choice != "11") {choice + ".csv"} else {scala.io.StdIn.readLine()}  // Reads user's input for data path if user enters '11'
		val dataset = Relation (filename, "dataset", null, -1) 			// Saving CSV as a relation
		val column_names = dataset.colNames			// Array of column names in relation
		val num_cols = dataset.cols					// Number of columns in dataset

		// Implementation for Mean Imputation 		
		for(i <- 0 to (num_cols - 1)){
			val selected = dataset.sigmaS(column_names(i), (x) => x!="")	// Filtering rows which have a missing value, as a no entry, i.e, ""
			val v_selected = selected.toVectorS(column_names(i))			// Converting remaining elements in column into a vector
			val v_seld = v_selected.map((x) => x.toDouble)					// Converting each element in filtered column to Double data type 
			val mean_col = (v_seld.sum) / selected.count(column_names(i))	// Computing mean of filtered column elements
			dataset.update(column_names(i), mean_col.toString(), "") 		// Updating blank spaces with mean of column
		} 
		
		// Giving user choice to execute model of their choice
		println("-"*75)
		println ("Select model:\n\t 1. Transformed Regression \n\t 2. Perceptron \n\t 3. NeuralNet_3L \n\t 4. NeuralNet_XL")
		println("-"*75)
		
		val model = scala.io.StdIn.readLine()
		if (model == "1") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			println("-"*75)
			println ("Select Transform Function:\n\t 1. log \n\t 2. sqrt \n\t 3. ~^2 \n\t 4. exp")
			println("-"*75)
			val function_choice = scala.io.StdIn.readLine()
			if (function_choice == "1"){
				tran_regression(x, y, transform_function = log, transform_inverse = exp)	// Implementing Transformed Regression Model with 'log' transform function
			}
			else if (function_choice == "2"){
				tran_regression(x, y, transform_function = sqrt , transform_inverse = sq)	// Implementing Transformed Regression Model with 'sqrt' transform function
			}
			else if (function_choice == "3"){
				tran_regression(x, y, transform_function = sq, transform_inverse = sqrt)	// Implementing Transformed Regression Model with 'sq' transform function
			}
			else if (function_choice == "4"){
				tran_regression(x, y, transform_function = exp, transform_inverse = log)	// Implementing Transformed Regression Model with 'exp' transform function
			}
			else {
				println("Invalid choice!")
			}
		}
		else if (model == "2") {
			val (x_initial, y_initial) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			val hp = new HyperParameter
			hp += ("eta", 0.1, 0.1)
			hp += ("bSize", 10, 10)
			hp += ("maxEpochs", 1000, 1000)

			println("-"*75)
			println ("Select Activation Function : \n\t 1. Identity \n\t 2. Sigmoid ")
			println("-"*75)
			val function_choice = scala.io.StdIn.readLine()
			if (function_choice == "1"){
			perceptron(x, y_initial, activation_f1 = f_id, hp)	// Implementing perceptron with 'id' activation function
			}
			else if (function_choice == "2"){
			perceptron(x, y_initial, activation_f1 = f_sigmoid, hp)	// Implementing perceptron with 'sigmoid' activation function
			}
			else {
				println("Invalid choice!")
			}
		}
		else if (model == "3") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			val xy = x :^+ y
			val hp = new HyperParameter
			hp += ("eta", 0.3, 0.3)
			hp += ("bSize", 10, 10)
			hp += ("maxEpochs", 2500, 2500)

			println("-"*75)
			println ("Select Activation Function:\n\t 1. Rectified Linear Unit \n\t 2. Leaky Rectified Linear Unit \n\t 3. Sigmoid ")
			println("-"*75)
			val function_choice = scala.io.StdIn.readLine()

			if (function_choice == "1"){
				neuralnet_3l(x, y, activation_f0 = f_reLU, hp)	// Implementing NeuralNet_3L with 'reLU' activation function
			}
			else if (function_choice == "2"){
				neuralnet_3l(x, y, activation_f0 = f_lreLU, hp)	// Implementing NeuralNet_3L with 'lreLU' activation function
			}
			else if (function_choice == "3"){
				neuralnet_3l(x, y, activation_f0 = f_sigmoid, hp)	// Implementing NeuralNet_3L with 'sigmoid' activation function
			}
			else {
				println("Invalid choice!")
			}
		}
		else if (model == "4") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			val hp = new HyperParameter
			hp += ("eta", 0.3, 0.3)
			hp += ("bSize", 10, 10)
			hp += ("maxEpochs", 1000, 1000)

			println("-"*75)
			println ("Select Activation Function for 1st Layer:\n\t 1. Rectified Linear Unit \n\t 2. Leaky Rectified Linear Unit \n\t 3. Sigmoid ")
			println("-"*75)
			val function_1_choice = scala.io.StdIn.readLine()

			println("-"*75)
			println ("Select Activation Function for 2nd Layer:\n\t 1. Rectified Linear Unit \n\t 2. Leaky Rectified Linear Unit \n\t 3. Sigmoid ")
			println("-"*75)
			val function_2_choice = scala.io.StdIn.readLine()

			if (function_1_choice == "1" && function_2_choice == "1"){
				neuralnet_xl(x, y, activation_f=  Array (f_reLU, f_reLU, f_lreLU), hp)	// Implementing NeuralNet_XL with 'reLU' and 'reLU' activation functions
			}
			else if (function_1_choice == "1" && function_2_choice == "2"){
				neuralnet_xl(x, y, activation_f=  Array (f_reLU, f_lreLU, f_lreLU), hp)	// Implementing NeuralNet_XL with 'reLU' and 'lreLU' activation functions
			}
			else if (function_1_choice == "1" && function_2_choice == "3"){
				neuralnet_xl(x, y, activation_f= Array (f_reLU, f_sigmoid, f_lreLU), hp)	// Implementing NeuralNet_XL with 'reLU' and 'sigmoid' activation functions
			}
			else if (function_1_choice == "2" && function_2_choice == "1"){
				neuralnet_xl(x, y, activation_f=  Array (f_lreLU, f_reLU, f_lreLU), hp)	// Implementing NeuralNet_XL with 'lrelu' and 'relu' activation function
			}
			else if (function_1_choice == "2" && function_2_choice == "2"){
				neuralnet_xl(x, y, activation_f=  Array (f_lreLU, f_lreLU, f_lreLU), hp)	// Implementing NeuralNet_XL with 'lrelu' and 'lrelu' activation function
			}
			else if (function_1_choice == "2" && function_2_choice == "3"){
				neuralnet_xl(x, y, activation_f=  Array (f_lreLU, f_sigmoid, f_lreLU), hp)	// Implementing NeuralNet_3L with 'lrelu' and 'sigmoid' activation function
			}
			else if (function_1_choice == "3" && function_2_choice == "1"){
				neuralnet_xl(x, y, activation_f=  Array (f_sigmoid, f_reLU, f_lreLU), hp)	// Implementing NeuralNet_3L with 'sigmoid' and 'relu' activation function
			}
			else if (function_1_choice == "3" && function_2_choice == "2"){
				neuralnet_xl(x, y, activation_f=  Array (f_sigmoid, f_lreLU, f_lreLU), hp)	// Implementing NeuralNet_3L with 'sigmoid' and 'lrelu' activation function
			}
			else if (function_1_choice == "3" && function_2_choice == "3"){
				neuralnet_xl(x, y, activation_f= Array (f_sigmoid, f_sigmoid, f_lreLU), hp)	// Implementing NeuralNet_3L with 'sigmoid' and 'sigmoid' activation function
			}
			else {
				println("Invalid choice!")
			}
		}
		else {
			println("Invalid choice!")
		}
	}
	main()
}
