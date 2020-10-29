name := "RTML"
version := "1.0"
scalaVersion := "2.11.12"
val spinalVersion = "1.4.2"

libraryDependencies ++= Seq(
  "com.github.spinalhdl" % "spinalhdl-core_2.11" % spinalVersion,
  "com.github.spinalhdl" % "spinalhdl-lib_2.11" % spinalVersion,
  "com.typesafe.play" %% "play-json" % "2.7.4",
  compilerPlugin("com.github.spinalhdl" % "spinalhdl-idsl-plugin_2.11" % spinalVersion)
)

fork := true
