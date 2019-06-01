package main

import (
	hnsw ".."
	"fmt"
	"github.com/360EntSecGroup-Skylar/excelize"
	"github.com/grd/stat"
	"math/rand"
	"strconv"
	"time"
)

var efSearch4 = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 150, 200, 300}

var NUM4, TESTNUM4, K4, DIMENSION4 int
var DIST4 string

func main() {
	preTypeList := []string{"sift1_4", "sift1_8", "sift1_16"}
	MList := []int{16, 32}
	efCList := []int{400, 800}
	
	NUM4 = 1000000
	TESTNUM4 = 100
	K4 = 10
	DIMENSION4 = 128
	DIST4 = "l2"

	xlsx := excelize.NewFile()

	for _, preType := range preTypeList {
		for l := range MList {
			fmt.Println("************************************")
			fmt.Printf("           %v: %v\n", preType, efCList[l])
			fmt.Println("************************************")

			p := make([]float32, DIMENSION4)
			h := hnsw.New(MList[l], efCList[l], p, DIST4)
			h.Grow(NUM4)
			sheetName := preType + "_" + strconv.FormatInt(int64(MList[l]), 10) + "_" + strconv.FormatInt(int64(efCList[l]), 10)
			xlsx.NewSheet(sheetName)
			_ = xlsx.SetCellValue(sheetName, "A1", "efSearch")
			_ = xlsx.SetCellValue(sheetName, "B1", "Query time(MS)")
			_ = xlsx.SetCellValue(sheetName, "C1", "Variance")
			_ = xlsx.SetCellValue(sheetName, "D1", "Precision")

			fmt.Println("Index loading: " + preType + "_" + strconv.FormatInt(int64(MList[l]), 10) + "_" + strconv.FormatInt(int64(efCList[l]), 10) + ".ind")
			h, timestamp, _ := hnsw.Load("ind/" + preType + "/" + preType + "_" + strconv.FormatInt(int64(MList[l]), 10) + "_" + strconv.FormatInt(int64(efCList[l]), 10) + ".ind")
			fmt.Printf("Index loaded, time saved was %v\n", time.Unix(timestamp, 0))

			fmt.Printf("Now searching with HNSW...\n")
			for iter, efs := range efSearch4 {
				timeRecord := make([]float64, TESTNUM4)
				hits := 0
				queries := make([]hnsw.Point, TESTNUM4)
				truth := make([][]uint32, TESTNUM4)
				for i := 0; i < TESTNUM4; i++ {
					searchAttr := []string{"blue", "sky", "boy"}
					//fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")

					queries[i] = randomPoint4()
					resultTruth := h.SearchBrute(queries[i], K4, searchAttr)
					truth[i] = make([]uint32, K4)
					for j := K4 - 1; j >= 0; j-- {
						item := resultTruth.Pop()
						truth[i][j] = item.ID
					}

					startSearch := time.Now()
					result := h.Search(queries[i], efs, K4, searchAttr)
					stopSearch := time.Since(startSearch)
					timeRecord[i] = stopSearch.Seconds() * 1000
					if result.Size != 0 {
						for j := 0; j < K4; j++ {
							item := result.Pop()
							//fmt.Printf("%v  ", item)
							if item != nil {
								//fmt.Println(h.GetNodeAttr(item.ID))
								for k := 0; k < K4; k++ {
									if item.ID == truth[i][k] {
										hits++
										break
									}
								}
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
				fmt.Printf("Average %v-NN precision: %v\n", K4, float64(hits)/(float64(TESTNUM4)*float64(K4)))
				fmt.Printf("\n")

				err := xlsx.SetCellValue(sheetName, "A"+fmt.Sprintf("%d", iter+2), efs)
				err = xlsx.SetCellValue(sheetName, "B"+fmt.Sprintf("%d", iter+2), decimal2(mean))
				err = xlsx.SetCellValue(sheetName, "C"+fmt.Sprintf("%d", iter+2), decimal2(variance))
				err = xlsx.SetCellValue(sheetName, "D"+fmt.Sprintf("%d", iter+2), decimal2(float64(hits)/(float64(TESTNUM4)*float64(K4))))
				if err != nil {
					panic(err)
				}

			}

			fmt.Println("----------------------------------")
			fmt.Printf(h.Stats())
			fmt.Println()
		}
	}
	err := xlsx.SaveAs("/Users/Ryan/Desktop/MA_imbalanced.xlsx")
	if err != nil {
		fmt.Println(err)
	}
}

func randomPoint4() hnsw.Point {
	var v hnsw.Point = make([]float32, DIMENSION4)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}

func decimal2(value float64) float64 {
	value, _ = strconv.ParseFloat(fmt.Sprintf("%.4f", value), 32)
	return value
}