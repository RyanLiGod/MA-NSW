package main

import (
	hnsw ".."
	"bufio"
	"fmt"
	"github.com/360EntSecGroup-Skylar/excelize"
	"github.com/grd/stat"
	"os"
	"strconv"
	"strings"
	"time"
)

type query2 struct {
	p    []float32
	attr []string
}

var efSearch3 = []int{10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 150, 200, 300}

var NUM3, TESTNUM3, K3, DIMENSION3 int
var DIST3 string

func main() {
	// preTypeList := []string{"siftsmall", "sift1_4", "sift1_8", "sift1_16", "glove25", "glove50", "glove100", "glove200"}
	//preTypeList := []string{"sift"}
	preTypeList := []string{"mnist"}
	MList := []int{16, 32}
	efCList := []int{400, 800}

	for _, preType := range preTypeList {
		for l := range MList {
			fmt.Println("************************************")
			fmt.Printf("           %v: %v\n", preType, efCList[l])
			fmt.Println("************************************")

			if preType == "siftsmall" {
				NUM3 = 10000
				TESTNUM3 = 100
				K3 = 10
				DIMENSION3 = 128
				DIST3 = "l2"
			} else if preType == "sift" || preType == "sift1_4" || preType == "sift1_8" || preType == "sift1_16" {
				NUM3 = 1000000
				TESTNUM3 = 10000
				K3 = 10
				DIMENSION3 = 128
				DIST3 = "l2"
			} else if preType == "gist" {
				NUM3 = 1000000
				TESTNUM3 = 1000
				K3 = 10
				DIMENSION3 = 960
			} else if preType == "glove25" {
				NUM3 = 1183514
				TESTNUM3 = 10000
				K3 = 10
				DIMENSION3 = 25
				DIST3 = "cosine"
			} else if preType == "glove50" {
				NUM3 = 1183514
				TESTNUM3 = 10000
				K3 = 10
				DIMENSION3 = 50
				DIST3 = "cosine"
			} else if preType == "glove100" {
				NUM3 = 1183514
				TESTNUM3 = 10000
				K3 = 10
				DIMENSION3 = 100
				DIST3 = "cosine"
			} else if preType == "glove200" {
				NUM3 = 1183514
				TESTNUM3 = 10000
				K3 = 10
				DIMENSION3 = 200
				DIST3 = "cosine"
			} else if preType == "mnist" {
				NUM3 = 60000
				TESTNUM3 = 10000
				K3 = 10
				DIMENSION3 = 784
				DIST3 = "l2"
			}

			xlsx, _ := excelize.OpenFile("../MA Experiments.xlsx")
			sheetName := preType + "_" + strconv.FormatInt(int64(MList[l]), 10) + "_" + strconv.FormatInt(int64(efCList[l]), 10)
			xlsx.NewSheet(sheetName)
			_ = xlsx.SetCellValue(sheetName, "A1", "efSearch")
			_ = xlsx.SetCellValue(sheetName, "B1", "Query time(MS)")
			_ = xlsx.SetCellValue(sheetName, "C1", "Variance")
			_ = xlsx.SetCellValue(sheetName, "D1", "Precision")

			prefix := "../dataset/" + preType + "_ma/" + preType
			queries := make([]query2, TESTNUM3)
			truth := make([][]uint32, TESTNUM3)

			queries = loadQueryData2(prefix)
			truth = loadGroundTruth(prefix)

			p := make([]float32, DIMENSION3)
			h := hnsw.New(MList[l], efCList[l], p, DIST3)
			h.Grow(NUM3)

			fmt.Println("Index loading: " + preType + "_" + strconv.FormatInt(int64(MList[l]), 10) + "_" + strconv.FormatInt(int64(efCList[l]), 10) + ".ind")
			h, timestamp, _ := hnsw.Load("ind/" + preType + "/" + preType + "_" + strconv.FormatInt(int64(MList[l]), 10) + "_" + strconv.FormatInt(int64(efCList[l]), 10) + ".ind")
			fmt.Printf("Index loaded, time saved was %v\n", time.Unix(timestamp, 0))

			fmt.Printf("Now searching with HNSW...\n")

			for iter, efs := range efSearch3 {
				timeRecord := make([]float64, TESTNUM3)
				hits := 0
				for i := 0; i < TESTNUM3; i++ {
					startSearch := time.Now()
					result := h.Search(queries[i].p, efs, K3, queries[i].attr)
					//fmt.Print("Searching with attributes:")
					//fmt.Println(attrQuery[i])
					stopSearch := time.Since(startSearch)
					timeRecord[i] = stopSearch.Seconds() * 1000
					if result.Size != 0 {
						for j := 0; j < K3; j++ {
							item := result.Pop()
							//fmt.Printf("%v  ", item)
							if item != nil {
								//fmt.Println(h.GetNodeAttr(item.ID))
								//var flag = 0
								for k := 0; k < K3; k++ {
									if item.ID == truth[i][k] {
										hits++
										//flag = 1
										break
									}
								}
								//if flag == 0 {
								//	fmt.Printf("Can't match: %v, i: %v, attr: %v", item.ID, i, queries[i].attr)
								//	fmt.Println()
								//}
							}
						}
					} else {
						fmt.Println("Can't return any node")
					}

					//fmt.Println()
				}

				data := stat.Float64Slice(timeRecord)
				mean := stat.Mean(data)
				variance := stat.Variance(data)

				fmt.Printf("--------------efs: %v---------------\n", efs)
				fmt.Printf("Mean of queries time(MS): %v\n", mean)
				fmt.Printf("Variance of queries time: %v\n", variance)
				fmt.Printf("%v queries / second (single thread)\n", 1000.0/mean)
				fmt.Printf("Average %v-NN precision: %v\n", K3, float64(hits)/(float64(TESTNUM3)*float64(K3)))
				fmt.Printf("\n")

				err := xlsx.SetCellValue(sheetName, "A"+fmt.Sprintf("%d", iter+2), efs)
				err = xlsx.SetCellValue(sheetName, "B"+fmt.Sprintf("%d", iter+2), decimal(mean))
				err = xlsx.SetCellValue(sheetName, "C"+fmt.Sprintf("%d", iter+2), decimal(variance))
				err = xlsx.SetCellValue(sheetName, "D"+fmt.Sprintf("%d", iter+2), decimal(float64(hits)/(float64(TESTNUM3)*float64(K3))))
				if err != nil {
					panic(err)
				}
			}
			err := xlsx.Save()
			if err != nil {
				fmt.Println(err)
			}
			fmt.Println("----------------------------------")
			fmt.Printf(h.Stats())
			fmt.Println()
		}
	}

}

func loadQueryData2(prefix string) []query2 {
	f, err := os.Open(prefix + "_query.txt")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 0
	queries := make([]query2, TESTNUM3)
	for s.Scan() {
		list := strings.Split(s.Text(), " ")
		vec := make([]float32, DIMENSION3)
		attr := list[DIMENSION3:]
		for i := 0; i < int(DIMENSION3); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		queries[count] = query2{p: vec, attr: attr}
		count++
		//if count%1000 == 0 {
		//	fmt.Printf("Read %v query records\n", count)
		//}
	}
	return queries
}

func loadGroundTruth(prefix string) [][]uint32 {
	f, err := os.Open(prefix + "_groundtruth.txt")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 0
	truth := make([][]uint32, TESTNUM3)
	for s.Scan() {
		list := strings.Split(s.Text(), " ")
		truthForOne := make([]uint32, K3)
		for i := 0; i < int(K3); i++ {
			v, _ := strconv.ParseUint(list[i], 10, 32)
			truthForOne[i] = uint32(v)
		}
		truth[count] = truthForOne
		count++
		//if count%1000 == 0 {
		//	fmt.Printf("Read %v truth records\n", count)
		//}
	}
	return truth
}

func decimal(value float64) float64 {
	value, _ = strconv.ParseFloat(fmt.Sprintf("%.4f", value), 32)
	return value
}
