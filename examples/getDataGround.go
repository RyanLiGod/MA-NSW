package main

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	hnsw ".."
	"github.com/grd/stat"
)

type job struct {
	p    []float32
	attr []string
	id   uint32
}

type query struct {
	p    []float32
	attr []string
}

const (
	M2              = 16
	efConstruction2 = 400
	efSearch2       = 200
)

var NUM2, TESTNUM2, K, DIMENSION2 int
var DIST2 string

func main() {
	// preType := "gist"
	// preType := "glove200"
	//preType := "mnist"
	//preType := "siftsmall"
	//preType := "sift"
	preType := "sift1_4"

	if preType == "siftsmall" {
		NUM2 = 10000
		TESTNUM2 = 100
		K = 100
		DIMENSION2 = 128
		DIST2 = "l2"
	} else if preType == "sift" || preType == "sift1_4" || preType == "sift1_8" || preType == "sift1_16" {
		NUM2 = 1000000
		TESTNUM2 = 10000
		K = 100
		DIMENSION2 = 128
		DIST2 = "l2"
	} else if preType == "gist" {
		NUM2 = 1000000
		TESTNUM2 = 1000
		K = 100
		DIMENSION2 = 960
		DIST2 = "l2"
	} else if preType == "glove25" {
		NUM2 = 1183514
		TESTNUM2 = 10000
		K = 100
		DIMENSION2 = 25
		DIST2 = "cosine"
	} else if preType == "glove50" {
		NUM2 = 1183514
		TESTNUM2 = 10000
		K = 100
		DIMENSION2 = 50
		DIST2 = "cosine"
	} else if preType == "glove100" {
		NUM2 = 1183514
		TESTNUM2 = 10000
		K = 100
		DIMENSION2 = 100
		DIST2 = "cosine"
	} else if preType == "glove200" {
		NUM2 = 1183514
		TESTNUM2 = 10000
		K = 100
		DIMENSION2 = 200
		DIST2 = "cosine"
	} else if preType == "mnist" {
		NUM2 = 60000
		TESTNUM2 = 10000
		K = 100
		DIMENSION2 = 784
		DIST2 = "l2"
	}

	prefix := "../dataset/" + preType + "_ma/" + preType

	points := make(chan job)
	queries := make(chan job)

	querySlice := make([]query, TESTNUM2)

	go loadBaseData(prefix, points)
	go loadQueryData(prefix, queries)

	p := make([]float32, DIMENSION2)
	h := hnsw.New(M2, efConstruction2, p, DIST2)
	h.Grow(NUM2)

	var wg sync.WaitGroup
	for i := 0; i < 1; i++ {
		wg.Add(1)
		go func() {
			for {
				job, more := <-points
				if !more {
					wg.Done()
					return
				}
				h.Add(job.p, job.id, job.attr)
			}
		}()
	}
	wg.Wait()

	err := h.Save("ind/" + preType + "/" + preType + "_" + strconv.FormatInt(M2, 10) + "_" + strconv.FormatInt(efConstruction2, 10) + ".ind")
	if err != nil {
		panic("Save error!")
	}

	h, timestamp, _ := hnsw.Load("ind/" + preType + "/" + preType + "_" + strconv.FormatInt(M2, 10) + "_" + strconv.FormatInt(efConstruction2, 10) + ".ind")
	fmt.Printf("Index loaded, time saved was %v\n", time.Unix(timestamp, 0))

	fmt.Printf("Now searching with HNSW...\n")
	timeRecord := make([]float64, TESTNUM2)
	hits := 0

	fmt.Printf("Generating queries and calculating true answers using bruteforce search...\n")

	type truthTyoe struct {
		sync.RWMutex
		p [][]uint32
	}

	var truth truthTyoe
	truth.p = make([][]uint32, TESTNUM2)

	var wg2 sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		wg2.Add(1)
		go func() {
			for {
				job, more := <-queries
				if !more {
					wg2.Done()
					return
				}
				result := h.SearchBrute(job.p, K, job.attr)
				truth.Lock()
				truth.p[int(job.id)] = make([]uint32, K)
				for j := K - 1; j >= 0; j-- {
					item := result.Pop()
					//fmt.Println(len(truth.p[int(job.id)]))

					truth.p[int(job.id)][j] = item.ID
				}
				truth.Unlock()
				//fmt.Println(truth)
				querySlice[int(job.id)] = query{p: job.p, attr: job.attr}
			}
		}()
	}
	wg2.Wait()

	// Save ground truth to file
	//var fileTruth = prefix + "_groundtruth.txt"
	////var fileTruth = "test" + "_groundtruth.txt"
	//f, _ := os.Create(fileTruth)
	//for _, line := range truth.p {
	//	for i, v := range line {
	//		_, _ = io.WriteString(f, strconv.FormatUint(uint64(v), 10))
	//		if i < len(line)-1 {
	//			_, _ = io.WriteString(f, " ")
	//		}
	//	}
	//	_, _ = io.WriteString(f, "\n")
	//}

	for i := 0; i < TESTNUM2; i++ {
		startSearch := time.Now()
		result := h.Search(querySlice[i].p, efSearch2, K, querySlice[i].attr)
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
					var flag = 0
					for k := 0; k < K; k++ {
						if item.ID == truth.p[i][k] {
							hits++
							flag = 1
							break
						}
					}
					if flag == 0 {
						fmt.Printf("Can't match: %v, i: %v, attr: %v", item.ID, i, querySlice[i].attr)
						fmt.Println()
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
	fmt.Printf("Average %v-NN precision: %v\n", K, float64(hits)/(float64(TESTNUM2)*float64(K)))
	fmt.Printf("\n")
	fmt.Printf(h.Stats())

}

func loadBaseData(prefix string, points chan job) {
	f, err := os.Open(prefix + "_base.txt")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 1
	for s.Scan() {
		list := strings.Split(s.Text(), " ")
		vec := make([]float32, DIMENSION2)
		attr := list[DIMENSION2:]
		for i := 0; i < int(DIMENSION2); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		points <- job{p: vec, attr: attr, id: uint32(count)}
		count++
		if count%1000 == 0 {
			fmt.Printf("Read %v records\n", count)
		}
	}
	close(points)
}

func loadQueryData(prefix string, queries chan job) {
	f, err := os.Open(prefix + "_query.txt")
	if err != nil {
		panic("couldn't open data file")
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	count := 0
	for s.Scan() {
		list := strings.Split(s.Text(), " ")
		vec := make([]float32, DIMENSION2)
		attr := list[DIMENSION2:]
		for i := 0; i < int(DIMENSION2); i++ {
			v, _ := strconv.ParseFloat(list[i], 32)
			vec[i] = float32(v)
		}
		queries <- job{p: vec, attr: attr, id: uint32(count)}
		count++
		if count%1000 == 0 {
			fmt.Printf("Read %v query records\n", count)
		}
	}
	close(queries)
	return
}
