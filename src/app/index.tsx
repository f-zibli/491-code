import React, { useEffect, useRef, useState, useCallback } from 'react';
import { AppState, ScrollView, Text, View, Button } from 'react-native';
import {
  initialize,
  getSdkStatus,
  SdkAvailabilityStatus,
  requestPermission,
  readRecords,
} from 'react-native-health-connect';

import type { RecordType } from 'react-native-health-connect';
import type { Permission } from 'react-native-health-connect';

const TYPES = [
  'HeartRate',
  'RespiratoryRate',
  'HeartRateVariabilityRmssd', // stress proxy
  'SleepSession',
  'ActiveCaloriesBurned',
  'Steps',
  'ExerciseSession',
  'OxygenSaturation',
];

const POLL_MS = 60 * 1000;

export default function App() {
  const [status, setStatus] = useState('Starting...');
  const [data, setData] = useState<Record<string, any>>({});
  const [lastSync, setLastSync] = useState<string | null>(null);
  const ready = useRef(false);

  const fetchAll = useCallback(async () => {
    if (!ready.current) return;
    // Look back 36h so last night's sleep is included
    const timeRangeFilter = {
      operator: 'between' as const,
      startTime: new Date(Date.now() - 36 * 3600 * 1000).toISOString(),
      endTime: new Date().toISOString(),
    };
    const result: Record<string, any> = {};
    for (const type of TYPES) {
      try {
        const { records } = await readRecords(type, { timeRangeFilter });
        result[type] = records;
      } catch (e: any) {
         result[type] = `error: ${e.message}`;
      }
    }
    setData(result);
    setLastSync(new Date().toLocaleTimeString());
    console.log(JSON.stringify(result, null, 2));
  }, []);

  // One-time setup: init + permissions
  useEffect(() => {
    (async () => {
      const sdk = await getSdkStatus();
      if (sdk !== SdkAvailabilityStatus.SDK_AVAILABLE) {
        setStatus('Health Connect not available');
        return;
      }
      await initialize();
      await requestPermission(
        TYPES.map((recordType) => ({ accessType: 'read', recordType })) as Permission[]
      );
      ready.current = true;
      setStatus('Connected');
      fetchAll();
    })().catch((e) => setStatus(`Setup error: ${e.message}`));
  }, [fetchAll]);

  // Automatic polling while the app is open
  useEffect(() => {
    const id = setInterval(fetchAll, POLL_MS);
    return () => clearInterval(id);
  }, [fetchAll]);

  // Refresh whenever the app returns to the foreground
  useEffect(() => {
    const sub = AppState.addEventListener('change', (s) => {
      if (s === 'active') fetchAll();
    });
    return () => sub.remove();
  }, [fetchAll]);

  return (
    <ScrollView style={{ flex: 1, paddingTop: 60, paddingHorizontal: 16 }}>
      <Text style={{ fontSize: 20, fontWeight: 'bold' }}>{status}</Text>
      <Text>Last sync: {lastSync ?? '-'}</Text>
      <Button title="Sync now" onPress={fetchAll} />
      {TYPES.map((t) => (
        <View key={t} style={{ marginTop: 12 }}>
          <Text style={{ fontWeight: 'bold' }}>
            {t}:{' '}
            {Array.isArray(data[t]) ? `${data[t].length} records` : data[t] ?? '-'}
          </Text>
        </View>
      ))}
    </ScrollView>
  );
}