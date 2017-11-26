name := "NlcTest"

version := "0.1"

scalaVersion := "2.11.11"

resolvers += "MavenRepository" at "https://mvnrepository.com/"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.2"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.2"