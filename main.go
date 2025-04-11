package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/hdf5"
)

// Structure pour représenter une entrée de données avant traitement de conversion en float64
type DataEntryRaw struct {
	C  string            `json:"c"`
	L  map[string]string `json:"l"`
	A  map[string]uint8  `json:"a"`
	La uint8             `json:"la"`
	V  [][]interface{}   `json:"v"`
}

// Structure pour représenter une entrée de données après traitement de conversion en float64
type DataEntryFloat struct {
	C  string            `json:"c"`
	L  map[string]string `json:"l"`
	A  map[string]uint8  `json:"a"`
	La uint8             `json:"la"`
	V  [][]float64       `json:"v"`
}

func main() {

	// Vérifier les arguments de la ligne de commande
	if len(os.Args) != 3 {
		fmt.Println("Usage: ./hdf5_test2 input.json output.h5")
		os.Exit(1)
	}

	inputFile := os.Args[1]
	outputFile := os.Args[2]

	// Lire le fichier JSON
	jsonData, err := os.ReadFile(inputFile)
	if err != nil {
		log.Fatalf("Erreur lors de la lecture du fichier JSON: %v", err)
	}

	// Décodage du JSON brut
	var rawdatasets [][]DataEntryRaw
	err = json.Unmarshal(jsonData, &rawdatasets)
	if err != nil {
		log.Fatalf("Erreur lors du décodage du JSON: %v", err)
	}

	// Prétraiter les données JSON pour convertir toutes les valeurs V en float64
	datasets := preprocessJsonData(rawdatasets)

	// Créer un fichier HDF5
	f, err := hdf5.CreateFile(outputFile, hdf5.F_ACC_TRUNC)
	if err != nil {
		log.Fatalf("Erreur lors de la création du fichier HDF5: %v", err)
	}
	defer f.Close()

	// Parcourir tous les datasets
	for _, dataset := range datasets {
		// Créer un groupe pour chaque dataset
		/*groupName := fmt.Sprintf("dataset_%d", datasetIndex)
		group, err := f.CreateGroup(groupName)
		if err != nil {
			log.Fatalf("Erreur lors de la création du groupe HDF5: %v", err)
		}
		defer group.Close()*/

		// Garder une trace des noms de datasets déjà utilisés
		datasetNames := make(map[string]int)

		// Parcourir toutes les entrées dans le dataset
		for _, entry := range dataset {
			// Vérifier qu'il y a des données à stocker
			if len(entry.V) == 0 {
				continue // Passer à l'entrée suivante si aucune donnée
			}

			// Vérifier si le nom existe déjà et générer un nom unique
			baseName := entry.C
			count, exists := datasetNames[baseName]

			var uniqueName string
			if exists {
				// Incrémenter le compteur et l'utiliser comme suffixe
				count++
				datasetNames[baseName] = count
				uniqueName = fmt.Sprintf("%s_%d", baseName, count)
			} else {
				// Premier dataset avec ce nom
				datasetNames[baseName] = 0
				uniqueName = baseName
			}

			// Déterminer les dimensions du dataset
			rows := len(entry.V)
			cols := len(entry.V[0])

			// Créer un espace pour le dataset
			dims := []uint{uint(rows), uint(cols)}
			space, err := hdf5.CreateSimpleDataspace(dims, nil)
			if err != nil {
				log.Fatalf("Erreur lors de la création de l'espace de données: %v", err)
			}
			defer space.Close()

			// Créer la propriété pour la compression
			prop, err := hdf5.NewPropList(hdf5.P_DATASET_CREATE)
			if err != nil {
				log.Fatalf("Erreur lors de la création de la liste de propriétés: %v", err)
			}
			defer prop.Close()

			// Configuer le chunking par colonne
			chunks := []uint{uint(rows), 1}

			// Configurer le chunking par ligne
			// chunks := []uint{1, uint(cols)}

			// Configurer le chunking sur la base de la taille de la matrice
			//chunks := []uint{uint(rows), uint(cols)}
			//if err := prop.SetChunk(chunks); err != nil {
			//log.Fatalf("Erreur lors de la configuration du chunking: %v", err)
			//}

			// Configurer le chunking de manière optimale
			/*var chunks []uint
			if rows > cols {
				chunks = []uint{uint(min(rows, 100)), uint(cols)} // chunk par blocs de lignes
			} else {
				chunks = []uint{uint(rows), uint(min(cols, 100))} // chunk par blocs de colonnes
			}*/

			if err := prop.SetChunk(chunks); err != nil {
				log.Fatalf("Erreur lors de la configuration du chunking: %v", err)
			}

			// Activer la compression GZIP (niveau 9)
			if err := prop.SetDeflate(9); err != nil {
				log.Printf("Avertissement: La compression GZIP n'a pas pu être activée: %v.", err)
			}

			// Créer un dataset directement avec le nom "c" de type float64
			dset, err := f.CreateDatasetWith(uniqueName, hdf5.T_NATIVE_DOUBLE, space, prop)
			if err != nil {
				log.Fatalf("Erreur lors de la création du dataset '%s': %v", entry.C, err)
			}
			defer dset.Close()

			// Pour garder une trace de l'association avec le nom original
			if err == nil && uniqueName != baseName {
				if err := addStringAttribute(dset, "original_name", baseName); err != nil {
					log.Printf("Erreur lors de l'ajout de l'attribut 'original_name': %v", err)
				}
			}

			// Ajout d'atributs pour le dataset s3p.activity
			if strings.Contains(entry.C, "s3p.activity") {
				entry.A["state_R"] = 1
				entry.A["state_r"] = 0
				entry.A["state_D"] = 7
				entry.A["state_d"] = 6
				entry.A["state_W"] = 5
				entry.A["state_w"] = 4
				entry.A["state_A"] = 3
				entry.A["state_a"] = 2
			}

			// Ajout d'attributs pour le dataset s3p.cruiseControlActive"
			if strings.Contains(entry.C, "s3p.cruiseControlActive") {
				entry.A["state_TRUE"] = 1
				entry.A["state_OFF"] = 0
			}

			// Ajout d'attributs pour le dataset s3p.ignition
			if strings.Contains(entry.C, "s3p.ignition") {
				entry.A["state_ON"] = 1
				entry.A["state_OFF"] = 0
			}

			// Pour "l" (convertir la map en attributs)
			for key, value := range entry.L {
				// Convertir la valeur en string pour simplification
				strValue := fmt.Sprintf("%v", value)
				if err := addStringAttribute(dset, "l_"+key, strValue); err != nil {
					log.Fatalf("Erreur lors de l'ajout de l'attribut 'l_%s': %v", key, err)
				}
			}

			// Pour "a" (attributs)
			for key, value := range entry.A {
				//strValue := fmt.Sprintf("%v", value)
				if err := addIntAttribute(dset, "a_"+key, value); err != nil {
					log.Fatalf("Erreur lors de l'ajout de l'attribut 'a_%s': %v", key, err)
				}
			}

			// Pour "la"
			if err := addIntAttribute(dset, "la", uint8(entry.La)); err != nil {
				log.Fatalf("Erreur lors de l'ajout de l'attribut 'la': %v", err)
			}

			// Convertir les données 2D en format plat pour HDF5
			flatData := make([]float64, rows*cols)
			for i := 0; i < rows; i++ {
				for j := 0; j < cols; j++ {
					flatData[i*cols+j] = entry.V[i][j]
				}
			}

			// Écrire les données
			err = dset.Write(&flatData)
			if err != nil {
				log.Fatalf("Erreur lors de l'écriture des données: %v", err)
			}

		}
	}

	fmt.Printf("Conversion réussie. Fichier HDF5 créé: %s\n", outputFile)
}

// Fonction auxiliaire pour ajouter un attribut de type string
func addStringAttribute(obj interface{}, name, value string) error {
	// Créer un type de données pour la chaîne
	dtype := hdf5.T_GO_STRING

	// Créer l'attribut
	dspace, err := hdf5.CreateSimpleDataspace([]uint{1}, nil)
	if err != nil {
		return err
	}

	var attr *hdf5.Attribute

	// Vérifier le type de l'objet
	switch o := obj.(type) {
	case *hdf5.Group:
		attr, err = o.CreateAttribute(name, dtype, dspace)
	case *hdf5.Dataset:
		attr, err = o.CreateAttribute(name, dtype, dspace)
	default:
		return fmt.Errorf("type d'objet non pris en charge pour les attributs")
	}

	if err != nil {
		return err
	}
	defer attr.Close()

	// Écrire la valeur
	return attr.Write(&value, dtype)
}

// Fonction auxiliaire pour ajouter un attribut de type int64
func addIntAttribute(obj interface{}, name string, value uint8) error {
	// Créer l'attribut
	dspace, err := hdf5.CreateSimpleDataspace([]uint{1}, nil)
	if err != nil {
		return err
	}

	var attr *hdf5.Attribute

	// Vérifier le type de l'objet
	switch o := obj.(type) {
	case *hdf5.Group:
		attr, err = o.CreateAttribute(name, hdf5.T_NATIVE_INT8, dspace)
	case *hdf5.Dataset:
		attr, err = o.CreateAttribute(name, hdf5.T_NATIVE_INT8, dspace)
	default:
		return fmt.Errorf("type d'objet non pris en charge pour les attributs")
	}

	if err != nil {
		return err
	}
	defer attr.Close()

	// Écrire la valeur
	return attr.Write(&value, hdf5.T_NATIVE_INT8)
}

// Fonction auxiliaire pour convertir les différents types de données en float64
func convertToFloat64(val interface{}, i, j int) float64 {
	switch v := val.(type) {
	case float64:
		return v
	case bool:
		if v {
			return 1.0
		}
		return 0.0
	/*case int64:
	return float64(v)*/
	case string:
		// Tenter de convertir la chaîne en nombre si possible
		if val == "true" || v == "TRUE" || v == "True" {
			return 1.0
		} else if val == "false" || v == "FALSE" || v == "False" {
			return 0.0
			// S3P.Activity
		} else if val == "R" { // Début de la période de repos "rest"
			return 1
		} else if val == "r" { // repos
			return 0
		} else if val == "D" { // Début de période de conduite "driving"
			return 7
		} else if val == "d" { // conduite
			return 6
		} else if val == "W" { // Début de la période de travail "working"
			return 5
		} else if val == "w" { // travail
			return 4
		} else if val == "A" { // Début de la période de disponibilité "available"
			return 3
		} else if val == "a" { // disponibilité
			return 2
			// S3P.Ignition
		} else if val == "ON" { // ignition on
			return 1
		} else if val == "OFF" { // ignition off
			return 0
		} else {
			// Tentative de conversion en float
			parsed, err := strconv.ParseFloat(v, 64)
			if err != nil {
				log.Printf("Avertissement: impossible de convertir la chaîne '%s' à [%d][%d] en nombre, utilisé 0.0", v, i, j)
				return 0 // valeur par défaut
			}
			return parsed
		}
	default:
		log.Printf("Avertissement: type non supporté à [%d][%d]: %T avec valeur %v, utilisé 0.0", i, j, val, val)
		return 0 // valeur par défaut
	}
}

// Fonction pour pré-traiter les données JSON et convertir toutes les valeurs V en float64
func preprocessJsonData(rawDatasets [][]DataEntryRaw) [][]DataEntryFloat {
	processedDatasets := make([][]DataEntryFloat, len(rawDatasets))

	for datasetIndex, rawDataset := range rawDatasets {
		processedDataset := make([]DataEntryFloat, len(rawDataset))

		for entryIndex, rawEntry := range rawDataset {
			// Créer une entrée avec les mêmes valeurs sauf pour V
			processedEntry := DataEntryFloat{
				C:  rawEntry.C,
				L:  rawEntry.L,
				A:  rawEntry.A,
				La: rawEntry.La,
			}

			// Traiter la matrice V
			if len(rawEntry.V) > 0 {
				rows := len(rawEntry.V)
				cols := len(rawEntry.V[0])

				processedV := make([][]float64, rows)
				for i := 0; i < rows; i++ {
					processedV[i] = make([]float64, cols)
					for j := 0; j < cols; j++ {
						if j < len(rawEntry.V[i]) { // Protection contre les lignes de longueurs différentes
							processedV[i][j] = convertToFloat64(rawEntry.V[i][j], i, j)
						}
					}
				}

				processedEntry.V = processedV
				// Inverser l'ordre des lignes
				//rows1 := len(processedEntry.V)
				for i := 0; i < rows/2; i++ {
					processedEntry.V[i], processedEntry.V[rows-i-1] = processedEntry.V[rows-i-1], processedEntry.V[i]
				}

				// Diviser les TS par 1000
				for i := 0; i < rows; i++ {
					for j := 0; j < cols; j++ {
						if j == 0 {
							processedEntry.V[i][j] = processedEntry.V[i][j] / 1000
						}
					}
				}
			}

			processedDataset[entryIndex] = processedEntry
		}

		processedDatasets[datasetIndex] = processedDataset
	}

	return processedDatasets
}
