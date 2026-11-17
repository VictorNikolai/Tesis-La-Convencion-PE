<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="tier_calidad" symbollevels="0" enableorderby="0">
    <categories>
      <category value="oro" symbol="0" label="Tier ORO (Alta Certeza &gt;90%)" render="true"/>
      <category value="plata" symbol="1" label="Tier PLATA (Certeza Media 70-90%)" render="true"/>
      <category value="bronce" symbol="2" label="Tier BRONCE (Apoyo)" render="true"/>
    </categories>
    <symbols>
      <symbol type="fill" name="0" alpha="0.65">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option type="QString" name="color" value="227,26,28,165"/>
            <Option type="QString" name="outline_color" value="140,0,0,255"/>
            <Option type="QString" name="outline_width" value="0.45"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="1" alpha="0.60">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option type="QString" name="color" value="255,127,0,150"/>
            <Option type="QString" name="outline_color" value="180,80,0,255"/>
            <Option type="QString" name="outline_width" value="0.35"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
          </Option>
        </layer>
      </symbol>
      <symbol type="fill" name="2" alpha="0.50">
        <layer class="SimpleFill" enabled="1" pass="0" locked="0">
          <Option type="Map">
            <Option type="QString" name="color" value="255,215,0,125"/>
            <Option type="QString" name="outline_color" value="160,120,0,255"/>
            <Option type="QString" name="outline_width" value="0.3"/>
            <Option type="QString" name="outline_width_unit" value="MM"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>
