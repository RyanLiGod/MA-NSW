package main

import (
	hnsw ".."
	"fmt"
	"github.com/grd/stat"
	"math/rand"
	"strconv"
	"time"
)

const (
	M4              = 32
	efConstruction4 = 800
)

var efSearch4 = []int{300}

var NUM4, TESTNUM4, K4, DIMENSION4 int
var DIST4 string

func main() {
	preType := []string{"sift1_4", "sift1_8", "sift1_16"}

	NUM4 = 1000000
	TESTNUM4 = 100
	K4 = 100
	DIMENSION4 = 128
	DIST4 = "l2"

	p := make([]float32, DIMENSION4)
	h := hnsw.New(M4, efConstruction4, p, DIST4)
	h.Grow(NUM4)

	for _, pre := range preType {
		fmt.Println("************************************")
		fmt.Printf("              %v\n", pre)
		fmt.Println("************************************")

		fmt.Println("Index loading: " + pre + "_" + strconv.FormatInt(M4, 10) + "_" + strconv.FormatInt(efConstruction4, 10) + ".ind")
		h, timestamp, _ := hnsw.Load("ind/" + pre + "/" + pre + "_" + strconv.FormatInt(M4, 10) + "_" + strconv.FormatInt(efConstruction4, 10) + ".ind")
		fmt.Printf("Index loaded, time saved was %v\n", time.Unix(timestamp, 0))

		fmt.Printf("Now searching with HNSW...\n")
		for _, efs := range efSearch4 {
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

		}

		fmt.Println("----------------------------------")
		fmt.Printf(h.Stats())
		fmt.Println()
	}

}

func randomPoint4() hnsw.Point {
	var v hnsw.Point = make([]float32, DIMENSION4)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
