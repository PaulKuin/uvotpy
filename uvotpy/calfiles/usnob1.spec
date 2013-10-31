# Catalog descriptor for USNO B1 over the web.
# Note that the StarID::SearchCat uses the scat WCS Tool.
type => StarID::SearchCat
fields => ID,RA,DEC,MAG,TYPE
packed => 0
data => Default
catalog/type => Indexed
catalog/n => 4

sort => m3
envvar => UB1_PATH
location => http://tdc-www.harvard.edu/cgi-bin/scat
# another USNO B1 server
#       http://archive.eso.org/skycat/servers/usnoa-server
limit => 3000
