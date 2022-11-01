from abc import abstractmethod
import spacy
import json, urllib # Needed libs
import urllib.request

class Linker(object):
    def __init__(self):
        self.path_to_ckpt = ''
        self.tokenizer = None
        self.model = None
        self.model_para = dict()
        self.model = self.instantiate()

    @abstractmethod
    def instantiate(self):
        pass
         
    def forward(self):
        pass


class RelationLinker(Linker):
    def __init__(self):
        super(RelationLinker, self).__init__()

    def instantiate(self):
        self.initWiki()
         
    def forward(self, text:str) -> list:
        return self.filter(self.Related(text))

    def entity2id(self, q):
	# Get wikidata id from wikidata api
        ans = []
        url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+"+".join(q.split(" "))+"&language=en"
        response = json.loads(urllib.request.urlopen(url).read())
        ans += response["search"]
        if (ans == [] and " " in q):
            # Reverse Trick : Pan Changjiang
            url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+"+".join(q.split(" ")[::-1])+"&language=en"
            response = json.loads(urllib.request.urlopen(url).read())
            ans += response["search"]
        if (ans == [] and len(q.split(" ")) > 2):
            # Abbreviation Trick
            url = "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search="+"+".join([q.split(" ")[0], q.split(" ")[-1]])+"&language=en"
            response = json.loads(urllib.request.urlopen(url).read())
            ans += response["search"]
        if len(ans) > 0:
            # Returns the first one, most likely one
            return ans[0]["id"]
        else:
            # Some outliers : Salvador Domingo Felipe Jacinto Dali i Domenech - Q5577
            return "Not Applicable"

    def getc(self, c):
        # Get entity name given entity id
        url = "https://www.wikidata.org/w/api.php?action=wbgetentities&props=labels&ids="+c+"&languages=en&format=json"
        response = json.loads(urllib.request.urlopen(url).read())
        return response["entities"][c]["labels"]["en"]["value"]

    def Related(self, name):
        # Get related property-entity (property id, property name, entity id, entity name) given entity name
        # Return a list of dicts, each dict contains (pid, property, eid, entity)
        # Fail to fetch eid would result in empty list
        query = self.entity2id(name)
        if query == "Not Applicable": return []
        ans = []
        url = "https://www.wikidata.org/w/api.php?action=wbgetentities&ids="+query+"&format=json&languages=en"
        response = json.loads(urllib.request.urlopen(url).read())
        for p in response["entities"][query]["claims"]:
            for c in response["entities"][query]["claims"][p]:
                # Enumerate property & entity (multi-property, multi-entity)
                try:
                    # Some properties are not related to entities, thus try & except
                    cid = c["mainsnak"]["datavalue"]["value"]["id"]
                    ans.append({
                        "pid": p,
                        "entity": self.getc(cid)
                        })
                    #ans.append("\\property\\"+p+"\t"+getp(p)+"\t\\entity\\"+cid+"\t"+getc(cid))
                    # Print in a pid-pname-eid-ename fashion
                except:
                    continue
        return ans

    def filter(self, ans, choose='P'):
        out = list()
        for ent in ans:
            if ent['pid'].startswith(choose):
                out.append({'ID':ent['pid'], 'label':ent['entity']})
        return out

    def initWiki(self,):
        # WARNING: RUN BEFORE USE GETP
        # Needed for property name fetching
        self.property_dict = {}
        url = "https://quarry.wmflabs.org/run/45013/output/1/json"
        # Fetch json from given lib
        res = json.loads(urllib.request.urlopen(url).read())
        for w in res["rows"]:
            self.property_dict[w[0]] = w[1]

class EntityLinker(Linker):
    def __init__(self):
        super(EntityLinker, self).__init__()

    def instantiate(self):
        model = spacy.blank("en")
        model.add_pipe('opentapioca')
        return model
         
    def forward(self, text:str) -> list:
        doc = self.model(text)
        out = list()
        for span in doc.ents:
            out.append((span.text, span.kb_id_, span.label_))
        return out 


if __name__ == "__main__":
    test = EntityLinker()
    test.instantiate()
    print(test.forward('Germany'))
    # test the relation linking module
    test = RelationLinker()
    test.instantiate()
    print(test.forward('border'))