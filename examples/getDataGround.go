package main

import (
	hnsw ".."
	"bufio"
	"fmt"
	"github.com/grd/stat"
	"io"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	//preType := "siftsmall"
	preType := "sift"

	const (
		//NUM     = 10000
		//TESTNUM = 100
		NUM            = 1000000
		TESTNUM        = 10000

		DIMENSION      = 128
		M              = 2
		efConstruction = 10
		efSearch       = TESTNUM
		K              = TESTNUM
	)

	prefix := preType + "_ma/" + preType

	f1, _ := os.Open(prefix + "_base.txt")
	f2, _ := os.Open(prefix + "_query.txt")

	s1 := bufio.NewScanner(f1)
	dataBase := make([][]float32, NUM)
	attrBase := make([][]string, NUM)
	dataCount := 0
	for s1.Scan() {
		list := strings.Split(s1.Text(), " ")
		vec := make([]float32, 128)
		attr := list[128:]
		for i := 0; i < int(DIMENSION); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		dataBase[dataCount] = vec
		attrBase[dataCount] = attr
		dataCount++
	}

	s2 := bufio.NewScanner(f2)
	dataQuery := make([][]float32, TESTNUM)
	attrQuery := make([][]string, TESTNUM)
	dataCount = 0
	for s2.Scan() {
		list := strings.Split(s2.Text(), " ")
		vec := make([]float32, 128)
		attr := list[128:]
		for i := 0; i < int(DIMENSION); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		dataQuery[dataCount] = vec
		attrQuery[dataCount] = attr
		dataCount++
	}

	var zero hnsw.Point = make([]float32, DIMENSION)
	h := hnsw.New(M, efConstruction, zero)
	h.Grow(NUM)

	for i := 1; i <= NUM; i++ {
		h.Add(dataBase[i-1], uint32(i), attrBase[i-1])
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}

	fmt.Printf("Now searching with HNSW...\n")
	timeRecord := make([]float64, TESTNUM)
	hits := 0
	for i := 0; i < TESTNUM; i++ {
		if (i)%1000 == 0 {
			fmt.Printf("Calculating using bruteforce search: %v\n", i)
		}
		//fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
		truth := make([][]uint32, TESTNUM)
		for i := range dataQuery {
			result := h.SearchBrute(dataQuery[i], K, attrQuery[i])
			truth[i] = make([]uint32, K)
			for j := K - 1; j >= 0; j-- {
				item := result.Pop()
				truth[i][j] = item.ID
			}
		}
		var fileTruth = prefix + "_groundtruth.txt"
		f, _ := os.Create(fileTruth)
		for _, line := range truth {
			for i, v := range line {
				_, _ = io.WriteString(f, strconv.FormatUint(uint64(v), 10))
				if i < len(line)-1 {
					_, _ = io.WriteString(f, " ")
				}
			}
			_, _ = io.WriteString(f, "\n")
		}
		startSearch := time.Now()
		result := h.Search(dataQuery[i], efSearch, K, attrQuery[i])
		//fmt.Print("Searching with attributes:")
		//fmt.Println(attrQuery[i])
		stopSearch := time.Since(startSearch)
		timeRecord[i] = stopSearch.Seconds() * 1000
		if result.Size != 0 {
			for j := 0; j < K; j++ {
				item := result.Pop()
				//fmt.Printf("%v  ", item)
				if item != nil {
					//fmt.Println(h.GetNodeAttr(item.ID))
					for k := 0; k < K; k++ {
						if item.ID == truth[i][k] {
							hits++
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

	fmt.Printf("Mean of queries time(MS): %v\n", mean)
	fmt.Printf("Variance of queries time: %v\n", variance)
	fmt.Printf("%v queries / second (single thread)\n", 1000.0/mean)
	fmt.Printf("Average 10-NN precision: %v\n", float64(hits)/(float64(TESTNUM)*float64(K)))
	fmt.Printf("\n")
	fmt.Printf(h.Stats())

}
