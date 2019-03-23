//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Aashish Yadavally
 *  @version 1.0
 *  @date    March 23, 2019
 *  @see     LICENSE (MIT style license file)
 */

package Perceptron

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The 'RegressionTest' object uses the defined cross-validation class, pre-defined 
* MatrixD and Regression classes to perform multiple regressions and subsequent analysis 
* on different numerical datasets, in the 'data' folder.  
*  > "sbt run" in the Scalation folder containing the build file to run the program.
* User gets two choices, once, to run on the dataset of his/her choice and again, to 
* choose the model to build the R2-RBar2-RCV2 graph on.
*/			
object PerceptronTest extends App {


	
	def main(){
		// Giving user the choice to select from ten datasets, or to give data path to own CSV file
		println("-"*75)
		println (" Select dataset: \n\t 1. Auto MPG \n\t 2. Beijing PM2.5 Dataset \n\t 3. Concrete Compressive Strength Dataset \n\t 4. Real Estate Valuation Dataset \n\t 5. Parkinson's Tele Monitoring \n\t 6. Computer Hardware")
		println("\t 7. Appliances Energy Prediction  \n\t 8. Combined Cycle Powerplant \n\t 9. CSM Dataset \n\t 10. Naval Propulsion Dataset \n\t 11. For other datasets, enter: /correct/path/to/data/csv")
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
		

	main()
}
