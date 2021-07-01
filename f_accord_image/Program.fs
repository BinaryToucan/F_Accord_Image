open System
open System.IO

open Accord.MachineLearning
open Accord.MachineLearning.VectorMachines
open Accord.MachineLearning.VectorMachines.Learning
open Accord.Statistics.Kernels
let training = @"C:/Users/binar/RiderProjects/f_accord_image/f_accord_image/bin/Debug/netcoreapp3.1/data/trainingsample.csv"
let validation = @"C:/Users/binar/RiderProjects/f_accord_image/f_accord_image/bin/Debug/netcoreapp3.1/data/validationsample.csv"

let readData filePath =
    File.ReadAllLines filePath
    |> fun lines -> lines.[1..]
    |> Array.map (fun line -> line.Split(','))
    |> Array.map (fun line -> 
        (line.[0] |> Convert.ToInt32), (line.[1..] |> Array.map Convert.ToDouble))
    |> Array.unzip

let labels, observations = readData training
let features = 28 * 28
let ``class`` = 10

let algorithm = 
    fun (svm: KernelSupportVectorMachine) 
        (classInputs: float[][]) 
        (classOutputs: int[]) (i: int) (j: int) -> 
        let strategy = SequentialMinimalOptimization(svm, classInputs, classOutputs)
        strategy :> ISupportVectorMachineLearning

let kernel = Linear()
let svm = new MulticlassSupportVectorMachine(features, kernel, ``class``)
let learner = MulticlassSupportVectorLearning(svm, observations, labels)
let config = SupportVectorMachineLearningConfigurationFunction(algorithm)
learner.Algorithm <- config

let error = learner.Run()
printfn "Error: %f" error

let validationLabels, validationObservations = readData validation

let outputs =
    Array.zip validationLabels validationObservations 
    |> Array.map (fun (l, o) -> if l = svm.Compute(o) then 1. else 0.)
    |> Array.average

printfn "Значение \t Предположнение "
let view =
    Array.zip validationLabels validationObservations 
    |> fun x -> x.[..5]
    |> Array.iter (fun (l, o) -> printfn "%i \t         %i" l (svm.Compute(o)))