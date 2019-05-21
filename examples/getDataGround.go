package main

import (
	hnsw ".."
	"bufio"
	"fmt"
	"github.com/grd/stat"
	"os"
	"strconv"
	"strings"
	"time"
)

var queries []job
//var truth [][]uint32

func main() {
	preType := "siftsmall"
	//preType := "sift"
	const (
		NUM      = 10000
		TESTNUM  = 100
		efSearch = 100
		K        = 100
		//NUM      = 1000000
		//TESTNUM  = 10000
		//efSearch = 1000
		//K        = 1000

		DIMENSION      = 128
		M              = 8
		efConstruction = 100
	)


	prefix := preType + "_ma/" + preType

	points := loadBaseData(prefix)
	queries = loadQueryData(prefix)

	var zero hnsw.Point = make([]float32, DIMENSION)
	h := hnsw.New(M, efConstruction, zero)
	h.Grow(NUM)

	for i := 1; i <= NUM; i++ {
		h.Add(points[i-1].p, uint32(i), points[i-1].attr)
		if (i)%1000 == 0 {
			fmt.Printf("%v points added\n", i)
		}
	}


	fmt.Printf("Now searching with HNSW...\n")
	timeRecord := make([]float64, TESTNUM)
	hits := 0

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")
	truth := make([][]uint32, len(queries))
	for i := range queries {
		//if (i)%10 == 0 {
		//	fmt.Printf("Calculating using bruteforce search: %v\n", i)
		//}
		result := h.SearchBrute(queries[i].p, K, queries[i].attr)
		truth[i] = make([]uint32, K)
		for j := K - 1; j >= 0; j-- {
			item := result.Pop()
			if item != nil {
				truth[i][j] = item.ID
			}
		}
	}

	//var fileTruth = prefix + "_groundtruth.txt"
	//f, _ := os.Create(fileTruth)
	//for _, line := range truth {
	//	for i, v := range line {
	//		_, _ = io.WriteString(f, strconv.FormatUint(uint64(v), 10))
	//		if i < len(line)-1 {
	//			_, _ = io.WriteString(f, " ")
	//		}
	//	}
	//	_, _ = io.WriteString(f, "\n")
	//}


	for i := 0; i < TESTNUM; i++ {
		startSearch := time.Now()
		result := h.Search(queries[i].p, efSearch, K, queries[i].attr)
		fmt.Print("Searching with attributes:")
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

		fmt.Println()
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

type job struct {
	p  []float32
	attr []string
	id uint32
}

func loadBaseData(prefix string) (points []job) {
	f, err := os.Open(prefix + "_base.txt")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 1
	points = make([]job, 10000)
	for s.Scan() {
		list := strings.Split(s.Text(), " ")
		vec := make([]float32, 128)
		attr := list[128:]
		for i := 0; i < int(128); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		points[count-1] = job{p: vec, attr: attr, id: uint32(count)}
		count++
		if count%1000 == 0 {
			fmt.Printf("Read %v records\n", count)
		}
	}
	return
}

func loadQueryData(prefix string) (queries []job) {
	f, err := os.Open(prefix + "_query.txt")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 0
	queries = make([]job, 10000)
	for s.Scan() {
		list := strings.Split(s.Text(), " ")
		vec := make([]float32, 128)
		attr := list[128:]
		for i := 0; i < int(128); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		queries[count] = job{p: vec, attr: attr, id: uint32(count)}
		count++
		if count%1000 == 0 {
			fmt.Printf("Read %v query records\n", count)
		}
	}
	return
}