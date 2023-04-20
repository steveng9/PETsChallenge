use okvs::hashable::Hashable;

#[derive(Debug, PartialEq, Eq)]
pub struct DataRow(pub Vec<String>);

impl Hashable for DataRow {
    fn to_bytes(&self) -> Vec<u8> {
        self.0.iter().flat_map(|x| x.as_bytes()).copied().collect()
    }
}
