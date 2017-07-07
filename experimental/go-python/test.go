package main

import "fmt"
import "github.com/sbinet/go-python"

func init() {
	err := python.Initialize()
	if err != nil {
		panic(err.Error())
	}
}

func main() {

	sysPath := python.PySys_GetObject("path")
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("./"))
	python.PyList_Insert(sysPath, 0, python.PyString_FromString("../../python/code"))

	/*numpy := python.PyImport_ImportModule("numpy")
	onesfunction := numpy.GetAttrString("ones")
	onesresult := onesfunction.CallFunction(python.PyInt_FromLong(5))

	pyByteArray := python.PyByteArray_FromObject(onesresult)
	goByteArray := python.PyByteArray_AsBytes(pyByteArray)

	fmt.Printf("%v\n", goByteArray)
	// Getting Golang string representation from Python string
	//fmt.Printf("%s\n", python.PyString_AsString(onesresult.Str()))

	custom := python.PyImport_ImportModule("custom")
	hwFunction := custom.GetAttrString("helloworld")
	hwResult := hwFunction.CallFunction()
	pyByteArray2 := python.PyByteArray_FromObject(hwResult)
	goByteArray2 := python.PyByteArray_AsBytes(pyByteArray2)

	fmt.Printf("%v\n", goByteArray2)

	incrFunction := custom.GetAttrString("incr")
	incrResult := incrFunction.CallFunction()
	fmt.Printf("%d\n", python.PyInt_AsLong(incrResult))

	incrResult = incrFunction.CallFunction()
	fmt.Printf("%d\n", python.PyInt_AsLong(incrResult))

	incrResult = incrFunction.CallFunction()
	fmt.Printf("%d\n", python.PyInt_AsLong(incrResult))

	err := python.PyRun_SimpleFile("../../python/code/logistic_main.py")
	if err != nil {
		fmt.Printf("%v\n", err)
	}*/

	i := 1000
	for i >= 0 {
		i = i - 1
		err := python.PyRun_SimpleFile("../../python/code/logistic_main.py")
		if err != nil {
			fmt.Printf("%v\n", err)
		}

	}

	//logisticMod := python.PyImport_ImportModule("logistic_main")
	//logisticFunc := logisticMod.GetAttrString("test")

	//i = 1000

	//for i >= 0 {
	//	logisticFunc.CallFunction()
	//}

	//logisticModule := python.PyImport_ImportModule("logistic_main")
	//logisticMain := logisticModule.GetAttrString("main")
	//logisticMain.CallFunction()

	/*for {
		incrResult = incrFunction.CallFunction()
		fmt.Printf("%d\n", python.PyInt_AsLong(incrResult))
	}*/
}
