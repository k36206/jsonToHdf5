package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"gonum.org/v1/hdf5"
)

// Structure pour représenter une entrée de données
type DataEntry struct {
	C  string                 `json:"c"`
	L  map[string]interface{} `json:"l"`
	A  map[string]interface{} `json:"a"`
	La int64                  `json:"la"`
	V  [][]float64            `json:"v"`
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

	// Décodage du JSON
	var datasets [][]DataEntry
	err = json.Unmarshal(jsonData, &datasets)
	if err != nil {
		log.Fatalf("Erreur lors du décodage du JSON: %v", err)
	}

	// Créer un fichier HDF5
	f, err := hdf5.CreateFile(outputFile, hdf5.F_ACC_TRUNC)
	if err != nil {
		log.Fatalf("Erreur lors de la création du fichier HDF5: %v", err)
	}
	defer f.Close()

	// Parcourir tous les datasets
	for datasetIndex, dataset := range datasets {
		// Créer un groupe pour chaque dataset
		groupName := fmt.Sprintf("dataset_%d", datasetIndex)
		group, err := f.CreateGroup(groupName)
		if err != nil {
			log.Fatalf("Erreur lors de la création du groupe HDF5: %v", err)
		}
		defer group.Close()

		// Parcourir toutes les entrées dans le dataset
		for _, entry := range dataset {
			// Créer un sous-groupe pour chaque entrée
			//entryGroupName := fmt.Sprintf("entry_%d", entryIndex)
			entryGroup, err := group.CreateGroup(entry.C)
			if err != nil {
				log.Fatalf("Erreur lors de la création du sous-groupe HDF5: %v", err)
			}
			defer entryGroup.Close()

			// Ajouter les métadonnées comme attributs
			// Pour "c"
			/*if err := addStringAttribute(entryGroup, "c", entry.C); err != nil {
				log.Fatalf("Erreur lors de l'ajout de l'attribut 'c': %v", err)
			}*/

			// Pour "l" (convertir la map en attributs)
			for key, value := range entry.L {
				// Convertir la valeur en string pour simplification
				strValue := fmt.Sprintf("%v", value)
				if err := addStringAttribute(entryGroup, "l_"+key, strValue); err != nil {
					log.Fatalf("Erreur lors de l'ajout de l'attribut 'l_%s': %v", key, err)
				}
			}

			// Pour "a" (attributs)
			for key, value := range entry.A {
				strValue := fmt.Sprintf("%v", value)
				if err := addStringAttribute(entryGroup, "a_"+key, strValue); err != nil {
					log.Fatalf("Erreur lors de l'ajout de l'attribut 'a_%s': %v", key, err)
				}
			}

			// Pour "la"
			if err := addIntAttribute(entryGroup, "la", int64(entry.La)); err != nil {
				log.Fatalf("Erreur lors de l'ajout de l'attribut 'la': %v", err)
			}

			// Créer un dataset pour les valeurs V (tableau 2D)
			if len(entry.V) > 0 {
				// Déterminer les dimensions du dataset
				rows := len(entry.V)
				cols := 2 // Vous avez mentionné que vous voulez un tableau de 2 colonnes

				// Créer un espace pour le dataset
				dims := []uint{uint(rows), uint(cols)}
				space, err := hdf5.CreateSimpleDataspace(dims, nil)
				if err != nil {
					log.Fatalf("Erreur lors de la création de l'espace de données: %v", err)
				}
				defer space.Close()

				// Créer le dataset
				dset, err := entryGroup.CreateDataset("values", hdf5.T_NATIVE_DOUBLE, space)
				if err != nil {
					log.Fatalf("Erreur lors de la création du dataset: %v", err)
				}
				defer dset.Close()

				// Convertir les données en format plat pour HDF5
				flatData := make([]float64, rows*cols)
				for i, row := range entry.V {
					for j, val := range row {
						if j < cols {
							flatData[i*cols+j] = val
						}
					}
				}

				// Écrire les données
				err = dset.Write(&flatData)
				if err != nil {
					log.Fatalf("Erreur lors de l'écriture des données: %v", err)
				}
			}
		}
	}

	fmt.Printf("Conversion réussie. Fichier HDF5 créé: %s\n", outputFile)
}

// Fonction auxiliaire pour ajouter un attribut de type string
func addStringAttribute(obj *hdf5.Group, name, value string) error {
	// Créer un type de données pour la chaîne
	dtype := hdf5.T_GO_STRING

	// Créer l'attribut
	dspace, err := hdf5.CreateSimpleDataspace([]uint{1}, nil)
	if err != nil {
		return err
	}
	attr, err := obj.CreateAttribute(name, dtype, dspace)
	if err != nil {
		return err
	}
	defer attr.Close()

	// Écrire la valeur
	return attr.Write(&value, dtype)
}

// Fonction auxiliaire pour ajouter un attribut de type int64
func addIntAttribute(obj *hdf5.Group, name string, value int64) error {
	// Créer l'attribut
	dspace, _ := hdf5.CreateSimpleDataspace([]uint{1}, nil)
	attr, err := obj.CreateAttribute(name, hdf5.T_NATIVE_INT64, dspace)
	if err != nil {
		return err
	}
	defer attr.Close()

	// Écrire la valeur
	return attr.Write(&value, hdf5.T_NATIVE_INT64)
}
