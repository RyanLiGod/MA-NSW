package main

import (
	hnsw ".."
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func main() {
	preType := "siftsmall"
	NUM := 10000
	DIMENSION := 128
	TESTNUM := 100
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
	dataQuery := make([][]float32, NUM)
	attrQuery := make([][]string, NUM)
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

	const (
		M              = 16
		efConstruction = 400
		K              = 100
	)

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

}
