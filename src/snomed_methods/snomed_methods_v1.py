# Copyright (c) 2026 SNOMED Methods Contributors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


@dataclass
class RetrieveSearchSynonymsConfig:
    filter_root_cui: int | str
    n_recursion: int = 10
    context_type: str = "xxxlong"
    type_id_filter: list[int] | None = None
    topn: int = 50
    debug: bool = False
    use_snomed: bool = True
    use_medcat: bool = True


@dataclass
class RetrieveSearchSynonymsMultiConfig:
    filter_root_cui_list: list[str]
    n_recursion: int = 10
    context_type: str = "xxxlong"
    type_id_filter: list[int] | None = None
    topn: int = 50
    debug: bool = False
    use_snomed: bool = True
    use_medcat: bool = True


@dataclass
class SnomedRelationsConfig:
    medcat: bool = False
    snowstorm: bool = False
    aliencat: bool = False
    dgx: bool = False
    dhcap: bool = False
    dhcap02: bool = True
    snomed_rf2_full_path: str | None = None
    medcat_path: str | None = None


@dataclass
class MedcatPathConfig:
    medcat_path: str | None
    aliencat: bool = False
    dgx: bool = False
    dhcap: bool = False
    dhcap02: bool = False


@dataclass
class SubsumedConceptsConfig:
    cui: int | str
    semantic_tags: list[str] | None = None
    max_depth: int = 10
    active_only: bool = True
    include_ancestors: bool = False
    include_descendants: bool = True


class SnomedRelations:
    def __init__(self, config: SnomedRelationsConfig) -> None:

        sys.path.insert(0, "..")

        snomed_default = os.environ.get(
            "SNOMED_RF2_PATH",
            "/data/snomed/SnomedCT_InternationalRF2_PRODUCTION_20231101T120000Z/Full/Terminology/sct2_StatedRelationship_Full_INT_20231101.txt",
        )

        if config.snomed_rf2_full_path is None:
            snomed_rf2_full_path = snomed_default
        else:
            snomed_rf2_full_path = config.snomed_rf2_full_path

        if not Path(snomed_rf2_full_path).exists():
            msg = (
                f"SNOMED RF2 path not found: {snomed_rf2_full_path}. "
                f"Set SNOMED_RF2_PATH environment variable to override the default."
            )
            raise FileNotFoundError(
                msg,
            )

        self.df: pd.DataFrame = pd.read_csv(snomed_rf2_full_path, sep="\t", header=0)

        self.medcat: bool = config.medcat

        self.snowstorm: bool = config.snowstorm

        if self.medcat:
            try:
                from medcat.cat import CAT  # noqa: PLC0415
            except ImportError:
                self.cat = None

            else:
                os.environ.get(
                    "MEDCAT_MODEL_PATH",
                    "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip",
                )
                medcat_path = self._get_medcat_path(
                    config.medcat_path,
                    MedcatPathConfig(
                        aliencat=config.aliencat,
                        dgx=config.dgx,
                        dhcap=config.dhcap,
                        dhcap02=config.dhcap02,
                    ),
                )
            try:
                self.cat = CAT.load_model_pack(medcat_path)
            except (FileNotFoundError, OSError):
                self.cat = None
        else:
            self.cat = None

    def _get_medcat_path(
        self,
        medcat_path: str | None,
        config: MedcatPathConfig,
    ) -> str:
        medcat_default = os.environ.get(
            "MEDCAT_MODEL_PATH",
            "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip",
        )
        if config.aliencat:
            return self._get_aliencat_path(medcat_path, medcat_default)
        if config.dgx:
            return self._get_dgx_path(medcat_path)
        if config.dhcap:
            return self._get_dhcap_path(medcat_path)
        if config.dhcap02:
            return self._get_dhcap02_path(medcat_path)
        if medcat_path is None:
            return medcat_default
        return medcat_path

    def _get_aliencat_path(
        self,
        medcat_path: str | None,
        medcat_default: str,
    ) -> str:
        medcat_path_env = os.environ.get("MEDCAT_ALIENCAT_PATH")
        if medcat_path is None:
            medcat_path = medcat_default
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dgx_path(self, medcat_path: str | None) -> str:
        medcat_path_env = os.environ.get("MEDCAT_DGX_PATH")
        if medcat_path is None:
            medcat_path = (
                "/data/medcat_models/20230328_trained_model_hfe_redone/"
                "medcat_model_pack_316666b47dfaac07"
            )
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dhcap_path(self, medcat_path: str | None) -> str:
        medcat_path_env = os.environ.get("MEDCAT_DHCAP_PATH")
        if medcat_path is None:
            medcat_path = "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip"
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def _get_dhcap02_path(self, medcat_path: str | None) -> str:
        medcat_path_env = os.environ.get("MEDCAT_DHCAP02_PATH")
        if medcat_path is None:
            medcat_path = "/data/medcat_models/medcat_model_pack_316666b47dfaac07.zip"
        if medcat_path_env:
            medcat_path = medcat_path_env
        return medcat_path

    def get_children(self, cui: int | str) -> list[int]:
        try:
            cui = int(cui)
            return self.df[self.df["destinationId"] == cui]["sourceId"].to_list()
        except ValueError:
            return []

    def get_parents(self, cui: int | str) -> list[int]:
        try:
            cui_int = int(cui)
            result = self.df[self.df["sourceId"] == cui_int]["destinationId"]
            return result.tolist()
        except ValueError:
            return []

    def expand_codes(
        self,
        filter_root_cui: int | str,
        *,
        debug: bool = False,
    ) -> tuple[list[int], list[str]]:

        if self.snowstorm:
            return self.expand_codes_snowstorm(filter_root_cui, debug=debug)

        return self.expand_codes_local(filter_root_cui, debug=debug)

    def expand_codes_local(
        self,
        filter_root_cui: int | str,
        *,
        debug: bool = False,
    ) -> tuple[list[int], list[str]]:

        if debug:
            pass

        retrieved_codes_temp = []
        retrieved_names_temp = []

        cr = self.expand_codes_parents_local(filter_root_cui, debug=debug)

        if debug:
            pass

        ar = self.expand_codes_children_local(filter_root_cui, debug=debug)

        if debug:
            pass

        retrieved_codes_temp.extend(cr[0])
        retrieved_codes_temp.extend(ar[0])
        retrieved_names_temp.extend(cr[1])
        retrieved_names_temp.extend(ar[1])
        retrieved_codes_temp = list(set(retrieved_codes_temp))
        retrieved_names_temp = list(set(retrieved_names_temp))

        if debug:
            pass

        return retrieved_codes_temp, retrieved_names_temp

    def has_medcat(self) -> bool:
        return (
            getattr(self, "medcat", False)
            and hasattr(self, "cat")
            and self.cat is not None
        )

    def get_pretty_name(self, cui: int | str) -> str | None:
        if not self.has_medcat():
            warnings.warn(
                f"MedCAT not available, returning None for cui: {cui}",
                stacklevel=2,
            )
            return None
        return self.cat.cdb.cui2preferred_name.get(str(cui))

    def get_pretty_name_list(
        self,
        cui_list: list[int | str],
    ) -> list[str | None]:
        if not self.has_medcat():
            return [None] * len(cui_list)

        pretty_name_list: list[str | None] = []

        for i in range(len(cui_list)):
            name = self.get_pretty_name(cui_list[i])
            if name is not None:
                pretty_name_list.append(name)
        return pretty_name_list

    def expand_codes_children_local(
        self,
        filter_root_cui: int | str,
        *,
        debug: bool = False,
    ) -> tuple[list[int], list[str]]:

        if debug:
            pass

        retrieved_names_temp = []

        children_codes = self.get_children(filter_root_cui)

        if self.medcat:
            retrieved_names_temp = self.get_pretty_name_list(children_codes)

        return children_codes, retrieved_names_temp

    def expand_codes_parents_local(
        self,
        filter_root_cui: int | str,
        *,
        debug: bool = False,
    ) -> tuple[list[int], list[str]]:

        if debug:
            pass

        retrieved_names_temp = []

        parent_codes = self.get_parents(filter_root_cui)

        if self.medcat:
            retrieved_names_temp = self.get_pretty_name_list(parent_codes)

        return parent_codes, retrieved_names_temp

    def expand_codes_snowstorm(
        self,
        filter_root_cui: int | str,
        *,
        debug: bool = False,
    ) -> tuple[list[int], list[str]]:
        if debug:
            pass

        retrieved_codes_temp: list[int] = []
        retrieved_names_temp: list[str] = []

        c = self.get_snowstorm_response_children(str(filter_root_cui))
        if debug:
            pass

        a = self.get_snowstorm_response_ancestors(str(filter_root_cui))
        if debug:
            pass

        cr = self.parse_snowstorm_response_to_cui_name(c)
        if debug:
            pass

        ar = self.parse_snowstorm_response_to_cui_name(a)
        if debug:
            pass

        retrieved_codes_temp.extend(cr[0])
        retrieved_codes_temp.extend(ar[0])
        retrieved_names_temp.extend(cr[1])
        retrieved_names_temp.extend(ar[1])
        retrieved_codes_temp = list(set(retrieved_codes_temp))
        retrieved_names_temp = list(set(retrieved_names_temp))

        if debug:
            pass

        return retrieved_codes_temp, retrieved_names_temp

    def recursive_code_expansion(
        self,
        filter_root_cui: int | str,
        n_recursion: int = 3,
        *,
        debug: bool = False,
    ) -> tuple[list[int], list[str]]:
        retrieved_codes: list[int | str] = [filter_root_cui]
        retrieved_names: list[str] = []

        for _i in tqdm(range(n_recursion)):
            if debug:
                pass

            retrieved_codes = list(set(retrieved_codes))

            for j in range(len(retrieved_codes)):
                current_filter_root_cui = retrieved_codes[j]

                raw = self.expand_codes_local(
                    filter_root_cui=current_filter_root_cui,
                    debug=debug,
                )

                codes = raw[0]
                names = raw[1]

                retrieved_codes.extend(codes)
                retrieved_names.extend(names)

                retrieved_codes = list(set(retrieved_codes))
                retrieved_names = list(set(retrieved_names))

        if debug:
            pass

        return retrieved_codes, retrieved_names

    def get_medcat_cdb_most_similar(
        self,
        cui: int | str,
        context_type: str = "xxxlong",
        type_id_filter: list[int] | None = None,
        topn: int = 10,
    ) -> tuple[list[str], list[str | None]]:
        if not self.has_medcat():
            warnings.warn("MedCAT not available, returning empty lists", stacklevel=2)
            return [], []
        if type_id_filter is None:
            type_id_filter = []
        try:
            res = self.cat.cdb.most_similar(
                cui,
                context_type=context_type,
                type_id_filter=type_id_filter,
                topn=topn,
            )
        except (ValueError, TypeError):
            return [], []

        names = []
        codes = []

        key_list = list(res.keys())

        codes = key_list.copy()

        names = self.get_pretty_name_list(codes)

        return codes, names

    def build_lists_medcat_snomedtree(
        self,
        input_list: list[int | str],
        *,
        medcat: bool = False,
        snomed: bool = True,
    ) -> tuple[list[int], list[str | None]]:
        retrieved_codes: list[int] = []
        retrieved_names: list[str] = []

        if snomed:
            for item in input_list:
                codes, names = self.recursive_code_expansion(item)
                retrieved_codes.extend(codes)
                retrieved_names.extend(names)

        if medcat:
            for item in input_list:
                codes, names = self.get_medcat_cdb_most_similar(item)
                retrieved_codes.extend([int(code) for code in codes])
                retrieved_names.extend(names)

        return retrieved_codes, retrieved_names

        # type-id t-16 etc
        # {T-38}    198890
        # {T-40}    173894
        # {T-11}     77284
        # {T-39}     64291
        # {T-18}     44201
        # {T-35}     34778
        # {T-6}      32944
        # {T-55}     27626
        # {T-33}     15117
        # {T-42}     14591

    def get_medcat_similar_score(
        self,
        input_cui: int | str,
        target_cui_list: list[int | str],
        *,
        debug: bool = False,
    ) -> list[float | None]:
        if not self.has_medcat():
            warnings.warn(
                "MedCAT not available, returning list of None values",
                stacklevel=2,
            )
            return [None] * len(target_cui_list)
        if debug:
            pass

        target_cui_list_str = list(map(str, target_cui_list))

        input_cui_str = str(input_cui)

        try:
            res = self.cat.cdb.most_similar(
                input_cui_str,
                context_type="xxxlong",
                type_id_filter=[],
                topn=999999,
            )
        except (ValueError, TypeError):
            return []

        results_list: list[float | None] = []

        if debug:
            pass

        for i in range(len(target_cui_list)):
            target_cui = target_cui_list_str[i]

            sim_res = res.get(target_cui)

            sim_score = sim_res.get("sim") if sim_res is not None else None

            if sim_score is not None:
                results_list.append(sim_score)
            else:
                results_list.append(np.nan)

        if debug:
            pass

        return results_list

    def append_concept_sim_to_df(
        self,
        df: pd.DataFrame,
        target_concept_sim_list: list[int | str],
        target_cui_list: list[int | str],
    ) -> pd.DataFrame:
        if not self.has_medcat():
            warnings.warn(
                "MedCAT not available, skipping append_concept_sim_to_df",
                stacklevel=2,
            )
            return df

        for elem in target_concept_sim_list:
            df[f"{elem}_concept_sim"] = self.get_medcat_similar_score(
                elem,
                target_cui_list,
                debug=True,
            )

        return df

    def retrieve_search_synonyms(
        self,
        config: RetrieveSearchSynonymsConfig,
    ) -> tuple[
        list[str],
        list[str],
        list[str],
        list[str | None],
        list[str],
    ]:
        # Initialize a list to store names
        if config.type_id_filter is None:
            config.type_id_filter = []
        all_names = []

        # Retrieve data for snomed_tree if use_snomed is True
        retrieved_codes_snomed_tree, retrieved_names_snomed_tree = (
            ([], [])
            if not config.use_snomed
            else self.recursive_code_expansion(
                filter_root_cui=config.filter_root_cui,
                n_recursion=config.n_recursion,
                debug=config.debug,
            )
        )

        # Add names to the list, stripping anything in parentheses
        all_names.extend(
            [
                re.sub(r"\([^)]*\)", "", name).strip()
                for name in retrieved_names_snomed_tree
                if name is not None
            ],
        )

        # Retrieve data for medcat_cdb if use_medcat is True
        retrieved_codes_medcat_cdb, retrieved_names_medcat_cdb = (
            ([], [])
            if not config.use_medcat
            else self.get_medcat_cdb_most_similar(
                config.filter_root_cui,
                context_type=config.context_type,
                type_id_filter=config.type_id_filter,
                topn=config.topn,
            )
        )

        # Add names to the list, stripping anything in parentheses
        all_names.extend(
            [
                re.sub(r"\([^)]*\)", "", name).strip()
                for name in retrieved_names_medcat_cdb
                if name is not None
            ],
        )

        return (
            retrieved_codes_snomed_tree,
            retrieved_names_snomed_tree,
            retrieved_codes_medcat_cdb,
            retrieved_names_medcat_cdb,
            all_names,
        )

    def retrieve_search_synonyms_multi(
        self,
        config: RetrieveSearchSynonymsMultiConfig,
    ) -> tuple[
        list[list[str]],
        list[list[str]],
        list[list[str]],
        list[list[str]],
        list[str],
        list[str],
    ]:
        """
        Retrieves search synonyms for multiple filter_root_cui values.

        Args:
            config (RetrieveSearchSynonymsMultiConfig): Configuration object containing
                all parameters for the search.

        Returns:
            Tuple[List[List[str]], List[List[str]], List[List[str]], List[List[str]],
                List[str], List[str]]: A tuple containing:
                - List of retrieved codes from SNOMED for each filter_root_cui.
                - List of retrieved names from SNOMED for each filter_root_cui.
                - List of retrieved codes from MedCAT for each filter_root_cui.
                - List of retrieved names from MedCAT for each filter_root_cui.
                - Flat list of all retrieved names.
                - Flat list of all retrieved codes.
        """
        # Initialize lists to store results
        if config.type_id_filter is None:
            config.type_id_filter = []
        all_retrieved_codes_snomed_tree = []
        all_retrieved_names_snomed_tree = []
        all_retrieved_codes_medcat_cdb = []
        all_retrieved_names_medcat_cdb = []
        all_names = []
        all_codes = []  # New list to store all codes

        # Iterate over each filter_root_cui in the list
        for filter_root_cui in config.filter_root_cui_list:
            # Retrieve data for snomed_tree if use_snomed is True
            retrieved_codes_snomed_tree, retrieved_names_snomed_tree = (
                ([], [])
                if not config.use_snomed
                else self.recursive_code_expansion(
                    filter_root_cui=filter_root_cui,
                    n_recursion=config.n_recursion,
                    debug=config.debug,
                )
            )

            # Add retrieved data to the lists
            all_retrieved_codes_snomed_tree.append(retrieved_codes_snomed_tree)
            all_retrieved_names_snomed_tree.append(retrieved_names_snomed_tree)

            # Add names to the all_names list, stripping anything in parentheses
            all_names.extend(
                [
                    re.sub(r"\([^)]*\)", "", name).strip()
                    for name in retrieved_names_snomed_tree
                    if name is not None
                ],
            )

            # Add codes to the all_codes list
            all_codes.extend(retrieved_codes_snomed_tree)

            # Retrieve data for medcat_cdb if use_medcat is True
            retrieved_codes_medcat_cdb, retrieved_names_medcat_cdb = (
                ([], [])
                if not config.use_medcat
                else self.get_medcat_cdb_most_similar(
                    filter_root_cui=filter_root_cui,
                    context_type=config.context_type,
                    type_id_filter=config.type_id_filter,
                    topn=config.topn,
                )
            )

            # Add retrieved data to the lists
            all_retrieved_codes_medcat_cdb.append(retrieved_codes_medcat_cdb)
            all_retrieved_names_medcat_cdb.append(retrieved_names_medcat_cdb)

            # Add names to the all_names list, stripping anything in parentheses
            all_names.extend(
                [
                    re.sub(r"\([^)]*\)", "", name).strip()
                    for name in retrieved_names_medcat_cdb
                    if name is not None
                ],
            )

            # Add codes to the all_codes list
            all_codes.extend(retrieved_codes_medcat_cdb)

            all_codes = [str(code) for code in all_codes]

        return (
            all_retrieved_codes_snomed_tree,
            all_retrieved_names_snomed_tree,
            all_retrieved_codes_medcat_cdb,
            all_retrieved_names_medcat_cdb,
            all_names,
            all_codes,
        )

    def get_snowstorm_response_children(
        self,
        concept_id: int | str,
    ) -> str | None:
        url = f"https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN%2FSNOMEDCT-GB/concepts/{concept_id}/children?form=inferred&includeDescendantCount=false"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            ),
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            return None
        else:
            return response.text

    def get_snowstorm_response_ancestors(
        self,
        concept_id: int | str,
    ) -> str | None:
        url = f"https://snowstorm.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN%2FSNOMEDCT-GB/concepts/{concept_id}/ancestors?excludeDirectChild=false"

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            ),
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            return None
        else:
            return response.text

    def parse_snowstorm_response_to_cui_name(
        self,
        response_text: str | None,
    ) -> tuple[list[str], list[str]]:
        if response_text is None:
            return [], []

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            return [], []

        codes: list[str] = []
        names: list[str] = []

        items = data.get("children", [])
        for item in items:
            concept_id = item.get("conceptId")
            if concept_id:
                codes.append(concept_id)
                preferred_term = item.get("preferredTerm", "")
                names.append(preferred_term)

        return codes, names

    def get_subsumed_concepts(
        self,
        config: SubsumedConceptsConfig,
    ) -> tuple[list[int], list[str]]:
        """Calculate transitive closure over is_a relationships to retrieve
        entire active DAG subgraph for a concept.

        Args:
            config (SubsumedConceptsConfig): Configuration object containing parameters.

        Returns:

        Returns:
            Tuple of (concept_ids, concept_names) lists.
        """
        try:
            cui = int(config.cui)
        except (ValueError, TypeError):
            return [], []
        if config.max_depth < 0:
            return [], []

        _, all_concepts, _ = self._subsumed_traverse(
            cui,
            max_depth=config.max_depth,
            active_only=config.active_only,
            include_ancestors=config.include_ancestors,
            include_descendants=config.include_descendants,
        )

        concept_ids = list(set(all_concepts))

        if config.semantic_tags and self.has_medcat():
            concept_ids = self._filter_by_semantic_tags(
                concept_ids,
                config.semantic_tags,
            )

        concept_names = self.get_pretty_name_list(concept_ids)

        return concept_ids, concept_names

    def _subsumed_traverse(
        self,
        cui: int,
        max_depth: int,
        *,
        active_only: bool,
        include_ancestors: bool,
        include_descendants: bool,
    ) -> tuple[set[int], list[int], dict[int, int]]:
        """Traverse the DAG and return visited concepts."""
        visited_concepts = set()
        all_concepts = [cui]
        depth_map = {cui: 0}

        def traverse(concept_id: int, depth: int) -> None:
            if depth > max_depth or concept_id in visited_concepts:
                return

            visited_concepts.add(concept_id)
            all_concepts.append(concept_id)
            depth_map[concept_id] = depth

            df = self._get_active_df() if active_only else self.df
            type_id = 116680003

            children_df = (
                df[(df["destinationId"] == concept_id) & (df["typeId"] == type_id)]
                if include_descendants
                else pd.DataFrame()
            )

            parents_df = (
                df[(df["sourceId"] == concept_id) & (df["typeId"] == type_id)]
                if include_ancestors
                else pd.DataFrame()
            )

            next_level_df = pd.concat([children_df, parents_df])

            if not next_level_df.empty:
                for _, row in next_level_df.iterrows():
                    # For children: destinationId is the parent, sourceId is the child
                    # For parents: sourceId is the child, destinationId is the parent
                    if "sourceId" in row and "destinationId" in row:
                        if include_descendants and row["destinationId"] == concept_id:
                            child_id = int(row["sourceId"])
                        elif include_ancestors and row["sourceId"] == concept_id:
                            child_id = int(row["destinationId"])
                        else:
                            child_id = (
                                int(row["sourceId"])
                                if include_descendants
                                else int(row["destinationId"])
                            )
                    else:
                        child_id = int(row.get("sourceId") or row.get("destinationId"))
                    traverse(child_id, depth + 1)

        traverse(cui, 0)
        return visited_concepts, all_concepts, depth_map

    def _get_active_df(self) -> pd.DataFrame:
        """Get active relationships dataframe."""
        if "active" in self.df.columns:
            return self.df[(self.df["active"] == "1") | (self.df["active"] == 1)]
        return self.df

    def _filter_by_semantic_tags(
        self,
        concept_ids: list[int],
        semantic_tags: list[str],
    ) -> list[int]:
        """Filter concepts by semantic tags."""
        filtered_ids = []
        for cid in concept_ids:
            name = self.get_pretty_name(cid)
            if name is not None:
                name_lower = name.lower()
                if any(tag.lower() in name_lower for tag in semantic_tags):
                    filtered_ids.append(cid)
        return filtered_ids
